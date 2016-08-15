

#' Spark ML - Binary Classification Evaluator 
#'
#' See the Spark ML documentation 
#'
#' @param predicted_tbl_spark The result of running sdf_predict
#' @param label A character string specifying which column contains the true, indexed labels (ie 0 / 1)
#' @param score A character string specifying which column contains the scored probability of a success (ie 1)
#' @param metric A character string specifying the metric: areaUnderRoc (default) or areaUnderPR
#'
#' @return  area under the specified curve
#' @export
#'

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
#' @param predicted_tbl_spark The result of running sdf_predict
#' @param label A string specifying the column that contains the true, indexed label. Support for binary and multi-class labels, column should be of double type (use as.double)
#' @param predicted_lbl A string specifying the column that contains the predicted label NOT the scored probability. Support for binary and multi-class labels, column should be of double type (use as.double)
#'
#' @return
#' @export
#'
#' @examples
ml_accuracy <- function(predicted_tbl_spark, label, predicted_lbl){
  df <- spark_dataframe(predicted_tbl_spark) 
  sc <- spark_connection(df)
  
  accuracy <- invoke_new(sc, "org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator") %>% 
    invoke("setLabelCol", label) %>% 
    invoke("setPredictionCol", predicted_lbl) %>% 
    invoke("setMetricName", "accuracy") %>% 
    invoke("evaluate", df)
  
  return(accuracy)
}


#' Spark ML - Feature Importance for Tree Models
#'
#' @param model An ml_model object, support for decision trees (>1.5.0), random forest (>2.0.0), GBT (>2.0.0)
#'
#' @return A sorted data frame with feature labels and their relative importance.
#' @export
#'
#' @examples
ml_tree_feature_importance <- function(model){
  supported <- c("ml_model_gradient_boosted_trees",
                 "ml_model_decision_tree",
                 "ml_model_random_forest")
  
  if ( !(class(model)[1] %in% supported)) {
    stop("Supported models include: ", paste(supported, collapse = ", "))
  }
  
  if (class(model) != "ml_model_decision_tree") spark_require_version(sc, "2.0.0")
  
  importance <- invoke(model$.model,"featureImportances") %>% 
    invoke("toArray") %>% 
    cbind(model$features) %>% 
    as.data.frame() 
  
  colnames(importance) <- c("importance", "feature")
  
  importance %>% arrange(desc(importance))
}
