library(sparklyr)
library(dplyr)
library(shiny)
library(tibble)

iris_tbl <- as_tibble(iris)

ui <- pageWithSidebar(
  headerPanel('Iris k-means clustering'),
  sidebarPanel(
    
    selectInput('xcol', 'X Variable', tbl_vars(iris_tbl)),
    selectInput('ycol', 'Y Variable', tbl_vars(iris_tbl),
                selected = tbl_vars(iris_tbl)[2]),
    numericInput('clusters', 'Cluster count', 3,
                 min = 1, max = 9)
  ),
  mainPanel(
    plotOutput('plot1')
  )
)

server <- function(input, output, session) {
  
  
  selectedData <- reactive({
    iris_tbl %>% select_(input$xcol, input$ycol)
  })
  
  
  clusters <- reactive({
    selectedData() %>%
      kmeans(centers = input$clusters)
  })
  
  output$plot1 <- renderPlot({
    par(mar = c(5.1, 4.1, 0, 1))
    
   
    #collect brings the data into R
    selectedData() %>% 
      plot(col = clusters()$cluster,
           pch = 20, cex = 4)
    
    points(clusters()$centers,
           pch = 4, cex = 4, lwd = 4)
  })
  
}

shinyApp(ui = ui, server = server)