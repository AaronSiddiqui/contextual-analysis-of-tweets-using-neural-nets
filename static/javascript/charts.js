// Bar chart
function createCharts(barData, radarData) {
    var myBarChart = new Chart(document.getElementById("barChart").getContext("2d"), {
        type: "bar",
        data: {
            labels: ["Negative", "Positive"],
            datasets: [{
                label: "Binary Sentiment Analysis",
                data: barData,
                backgroundColor: [
                    "rgba(255, 99, 132, 0.2)",
                    "rgba(0, 204, 0, 0.2)"
                ],
                borderColor: [
                    "rgba(255,99,132,1)",
                    "rgba(0, 204, 0, 1)"
                ],
                borderWidth: 1
            }]
        },
        options: {
            legend: {
                labels: {
                    fontColor: "rgba(240, 173, 78)",
                    fontSize: 15
                }
            },
            scales: {
                xAxes: [{
                    gridLines: {
                        color: "rgba(240, 173, 78, 0.2)"
                    },
                    ticks: {
                        fontColor: "rgba(240, 173, 78)"
                    }
                }],
                yAxes: [{
                    gridLines: {
                        color: "rgba(240, 173, 78, 0.2)"
                    },
                    ticks: {
                        beginAtZero: true,
                        fontColor: "rgba(240, 173, 78)"
                    }
                }]
            },
            responsive: false
        }
    });

    // Radar chart
    var myRadarChart = new Chart(document.getElementById("radarChart").getContext("2d"), {
        type: "radar",
        data: {
            labels: ["Joy", "Fear", "Anger", "Sadness", "Disgust", "Shame", "Guilt"],
            datasets: [{
                label: "Emotion Detection",
                data: radarData,
                backgroundColor: [
                    "rgb(0, 204, 204, 0.2)"
                ],
                borderColor: [
                    "rgb(0, 204, 204)"
                ],
                borderWidth: 1
            }]
        },
        options: {
            legend: {
                labels: {
                    fontColor: "rgba(240, 173, 78)",
                    fontSize: 15
                }
            },
            scale: {
                gridLines: {
                    color: "rgba(240, 173, 78, 0.2)"
                },
                angleLines: {
                    color: "rgba(240, 173, 78, 0.2)"
                },
                pointLabels: {
                    fontColor: "rgba(240, 173, 78)"
                },
            },
            responsive: false
        }
    });
}

// function updateChartData(barData, radarData) {
//     this.barData = barData;
//     this.radarData = radarData;
//     console.log(this.barData)
//     myBarChart.update();
// }
