library(sparklyr)
library(dplyr)
library(shiny)


#Connect to Spark
# Sys.setenv(SPARK_HOME = "...") TODO
sc <- spark_connect(master = "local")

#Read in Parquet Data
spark_read_parquet(sc, "iris", "shiny_iris_data")
iris_tbl <- tbl(sc, "iris")
cols <- glimpse(iris_tbl) %>% 
  colnames()

ui <- pageWithSidebar(
  headerPanel('Iris k-means clustering'),
  sidebarPanel(
    
    selectInput('xcol', 'X Variable', cols),
    selectInput('ycol', 'Y Variable', cols,
                selected = cols[2]),
    numericInput('clusters', 'Cluster count', 3,
                 min = 1, max = 9)
  ),
  mainPanel(
    plotOutput('plot1')
  )
)

server <- function(input, output, session) {
  
  # Nothing is evaluated in Spark at this step
  selectedData <- reactive({
    iris_tbl %>% select_(input$xcol, input$ycol)
  })
  
  # The Spark data frame is constructed and kmeans is run
  clusters <- reactive({
    selectedData() %>%
      ml_kmeans(centers = input$clusters)
  })
  
  output$plot1 <- renderPlot({
    par(mar = c(5.1, 4.1, 0, 1))
    
    #score the results in Spark, pull in results to R
    scored <- predict(clusters(), iris_tbl) + 1
    
    #collect brings the data into R
    selectedData() %>% 
      collect() %>% 
      plot(col = scored,
           pch = 20, cex = 4)
    
    points(clusters()$centers,
           pch = 4, cex = 4, lwd = 4)
  })
  
}

shinyApp(ui = ui, server = server)