#' Spark ML - Feature Importance for Tree Models
#'
#' @param sc An active spark context
#' @param model An ml_model object of class decision trees (>1.5.0), random forest (>2.0.0), or GBT (>2.0.0)
#'
#' @return A sorted data frame with feature labels and their relative importance.
#' @export


ml_tree_feature_importance <- function(sc, model){
  supported <- c("ml_model_gradient_boosted_trees",
                 "ml_model_decision_tree",
                 "ml_model_random_forest")
  
  if ( !(class(model)[1] %in% supported)) {
    stop("Supported models include: ", paste(supported, collapse = ", "))
  }
  
  if (class(model)[1] != "ml_model_decision_tree") spark_require_version(sc, "2.0.0")
  
  importance <- invoke(model$.model,"featureImportances") %>%
    invoke("toArray") %>%
    cbind(model$features) %>%
    as.data.frame()
  
  colnames(importance) <- c("importance", "feature")
  
  importance %>% dplyr::arrange(dplyr::desc(importance))
}




#' Spark ML - Binary Classification Evaluator
#'
#' See the Spark ML Documentation \href{https://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.evaluation.BinaryClassificationEvaluator}{BinaryClassificationEvaluator}
#'
#' @param predicted_tbl_spark The result of running sdf_predict
#' @param label Name of column string specifying which column contains the true, indexed labels (ie 0 / 1)
#' @param score Name of column contains the scored probability of a success (ie 1)
#' @param metric The classification metric -  one of: areaUnderRoc (default) or areaUnderPR
#'
#' @return  area under the specified curve
#' @export


ml_binary_classification_eval <- function(predicted_tbl_spark, label, score, metric = "areaUnderROC"){
  df <- spark_dataframe(predicted_tbl_spark)
  sc <- spark_connection(df)
  
  envir <- new.env(parent = emptyenv())
  
  tdf <- df %>%
    ml_prepare_dataframe(response = label, feature = c(score, score), envir = envir)
  
  auc <- invoke_new(sc, "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator") %>%
    invoke("setLabelCol", envir$response) %>%
    invoke("setRawPredictionCol", envir$features) %>%
    invoke("setMetricName", metric) %>%
    invoke("evaluate", tdf)
  
  return(auc)
}

#' Spark ML - Classification Evaluator
#'
#' See the Spark ML Documentation \href{https://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator}{MulticlassClassificationEvaluator}
#'
#' @param predicted_tbl_spark A tbl_spark object that contains a columns with predicted labels
#' @param label Name of the column that contains the true, indexed label. Support for binary and multi-class labels, column should be of double type (use as.double)
#' @param predicted_lbl Name of the column that contains the predicted label NOT the scored probability. Support for binary and multi-class labels, column should be of double type (use as.double)
#' @param metric A classification metric, one of: f1 (default), precision, recall, weightedPrecision, weightedRecall, accuracy
#'
#' @return see \code{metric}
#'
#' @export


ml_classification_eval <- function(predicted_tbl_spark, label, predicted_lbl, metric = "f1"){
  df <- spark_dataframe(predicted_tbl_spark)
  sc <- spark_connection(df)
  
  res <- invoke_new(sc, "org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator") %>%
    invoke("setLabelCol", label) %>%
    invoke("setPredictionCol", predicted_lbl) %>%
    invoke("setMetricName", metric) %>%
    invoke("evaluate", df)
  
  return(res)
}


