---
title: "Predictive maintenance: A time series approach"
date: 2019-01-30
tags: [Time Series Modelling, Prognostics]
header:
  image: "/images/predmaintenance/cover.jpg"
mathjax: "true"
---

Unexpected downtime has a significant effect on throughput in manufacturing. Managing the service life of equipment helps in reducing downtime costs. The ability to predict equipment outage helps in deploying pre-failure maintenance and bring down unplanned downtime costs. Quite commonly, these machines produce streams of time series data which can be modeled using markovian techniques. In this post we look at using time series techniques to forecast the failure of such machines.  
In particular we'll be using ARIMA, exponential smoothing (Holt-Winter's) and a single layer perceptron model on the C-MAPSS dataset from NASA's prognostics data repository. You can find the dataset [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## Introduction

First, let us look at the dataset characteristics:

Data is a from a fleet of 4 similar simulated aircraft engines operating under different conditions. Data is available from 21 internal sensors and 3 external sensors. This data is contaminated with sensor noise where in each case the engine is operating normally at the start of each time series and develops a fault at some point during the series and eventually deteriorates until failure. Each dataset is divided into:

1. Training: In the training set, the fault grows in magnitude until system failure.
2. Test: In the test set, the time series ends some time prior to system failure.
3. RUL vector: Delta of (actual failure - last sensor observation) from the test set.
4.
Clearly, these datasets were geared towards data-driven approaches where very little or no system information was made available. The fundamental difference between these 4 datasets is attributed to the number of simultaneous fault modes and the operational conditions simulated in these experiments. In this post, we will be working with the first simulated dataset. Note that the first dataset has 100 different engines within it. Below is a visual interpretation of the dataset:

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/dataset.jpg" alt="Visual outlook of the dataset charecterstics">

Our objective is to forecast the RUL of the test set i.e the red colored portion in the above image. 

## Preprocessing:

**1. Sensor selection:**

<figure>
  <img src="{{site.url}}/images/predmaintenance/all_sensors.jpg" alt="my alt text"/>
  <figcaption>Plots of 21 sensor measurements</figcaption>
</figure>


The scatter plots above give us an intuition regarding the health of the engine w.r.t to various sensors. However, not all sensors are equally important as some of them do not provide any information and others provide conflicting evidence. Therefore, based on the sensor patterns, we classify them into three categories: 1) Continuous and consistent 2) Discrete and 3) Continuous and Inconsistent.

Sensors that have one or more discrete values do not increase our understanding of the engine health hence can be eliminated. Sensors that have continuous but inconsistent values may contain some hidden information but due to inconsistencies towards the end they tend to be rather misleading and therefore need to be eliminated as well. Only sensors that have continuous and consistent values are chosen i.e (sensors 2, 3, 4, 7, 8,11, 12, 13, 15, 20 and 21). These sensors will aid in modelling a mathematical function to describe engine degradation.

<figure>
  <img src="{{site.url}}/images/predmaintenance/op.jpg" alt="my alt text"/>
  <figcaption>Plots of the 3 operational settings</figcaption>
</figure>

All three operational settings are ignored based on the sensor selection criteria. 

**2. Health Index:**

The intuition behind using a health index is to model a univariate time series, as a function of the selected sensors, that will help us understand the degradation pattern of each engine. 

In order to do so, first we would need a target variable w.r.t to which various sensor inputs can be modelled. As we do not have a target variable we will create one for each of the 100 engines, in the training set, by selecting the first 30 cycles and labelling them as 1 (assuming they represent a healthy condition) and the last 30 cycles as 0 (assuming they represent a deteriorated condition). 

You could probably ask why 30? 

So, first, we first looked at the summary of max RUL

```r
#Look at the RUL summary
RUL_engine = aggregate(RUL~Engine, data, max)
summary(RUL_engine$RUL)
```

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/rul_summary.jpg" alt="Max RUL summary">

which gave us 128 as the minimum value. Now, in these 128 cycles naturally first few are good and the last few are bad as the engine starts well and then eventually deteriorates. To give us some breathing space and to clearly distinguish between good and bad we divide by 2 to attain 64. Now, in these 64 we can assign the first half as 1 and the last half as 0 but for better symmetry we choose 30. 

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/hi1.jpg">

<figure>
  <img src="{{site.url}}/images/predmaintenance/hi2.jpg" alt="my alt text"/>
  <figcaption>HI for engine 1 in the training set</figcaption>
</figure>

After assigning 0's /1's, removing null values and keeping only the required sensors. We look at the correlation between the remaining sensors and the manually curated health index.

<figure>
  <img src="{{site.url}}/images/predmaintenance/corr.jpg" alt="my alt text"/>
  <figcaption>Correlation plot between continuous - consistent sensors and health index</figcaption>
</figure>

As we can observe from the above correlation plot, all the sensors have a significant correlation with the HI. Therefore, we will proceed without eliminating any of them.

Now, we will use our target i.e HI and features i.e continuous and consistent sensors, to create a model which will give us the HI on the test set. We will employ a linear regression to do so. 

$$
\begin{aligned}

Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + ... + \beta_k x_k + \epsilon

\end{aligned}
$$

The above regression equation gives us estimated parameters. Using these regression parameters on our test set, we will obtain an HI for all the 100 engines in the test set. The HI for each of those engine will be a univariate time series which can be modeled using Time series techniques.

```r
#Using regression on the training set to obtain a health index
regression = lm(HI~., data = data_final)
```

This model stored in *regression* can now be used on the test set to create a univariate time series aka a health index that will help us understand the degradation pattern of each engine.

```r
#reading in the test file
test = read.csv("test.csv")
test_s = subset(test, select = c("S2","S3","S4",
                                   "S7","S8","S11"
                                   ,"S12","S13","S15",
                                   "S20","S21"))
colnames(test_s) = c("Sensor2","Sensor3","Sensor4",
                      "Sensor7","Sensor8","Sensor11",
                      "Sensor12","Sensor13","Sensor15",
                      "Sensor20","Sensor21")
test_model = predict.lm(regression, test_s) #using the regression parameters to get HI
test_s$HI = test_model #Found HI
test = cbind(test, test_s$HI) #Cbind HI with original test
```
The below code will plot HI for all the 100 engines

```r
#creating a function to map engine to the ts.plot function
plot_hi = function(j){
  test_1 = test[test$Engine == j,]
  case1 = test_1$`test_s$HI`
  ts.plot(case1, col = 'lightpink3', xlab = 'Cycle', ylab = 'Health Index')
}

engines = 1:100

hilist = list()
for (j in engines){
  hilist[[j]] = plot_hi(j)
}
```
Attaching HI for 5 of the 100 engines:

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/one.jpg" alt="Correlation plot between continuous - consistent sensors and health index">

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/two.jpg" alt="Correlation plot between continuous - consistent sensors and health index">

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/three.jpg" alt="Correlation plot between continuous - consistent sensors and health index">

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/four.jpg" alt="Correlation plot between continuous - consistent sensors and health index">

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/five.jpg" alt="Correlation plot between continuous - consistent sensors and health index">

This marks the end of preprocessing.

## Modelling

We'll .....

### ARIMA

ARIMA is a classic statistical approach to model univariate data. An ideal approach in using ARIMA(p,d,q) is to:

1. Plot the time series to check if it is stationary:

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/arima_1.jpg" alt="something">

2. If the series is not stationary, make it by differencing (to remove trend) or apply a power transformation (to stablize variance): <- this is the d part:

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/arima_2.jpg" alt="something">

3. Plot the ACF/PACF of the stationary series to identify AR/MA order. AR or MA terms are needed to correct any autocorrelation that remains in the differenced series. The partial autocorrelation at lag k is equal to the estimated AR(k) coefficient in an autoregressive model with k terms and the lag at which the ACF cuts off is the indicated number of MA terms: <- this is the p,q part

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/arima_3.jpg" alt="something">

4. Supply the parameters p and q to the model. $$p = 2, q = 2$$ in this case. Note that we pick p = 2 instead of p = 4 for a more parsimonious model.

5. Analyse residuals to make sure the fit is good and doesn't violate any of the assumptions:

<img src="{{ site.url }}{{ site.baseurl }}//images/predmaintenance/arima_4.jpg" alt="something">

Analyzing our residuals,
- Normality assumption is met based on the Q-Q plot.
- Residuals look indepent identically distributed.
- As our p-value is greater than 0.5 in the Ljung–Box–Pierce Q-statistic it indicates that our ACF residuls follow a $X^2$ distribution. Note that the null hypothesis, $$H_0$$ here is that our residuals do not follow a chi-squared distribution.

But as we would like to forecast RUL for multiple engines, doing this for each engine is not effective. Therefore, we will use the auto.arima() function from the forecast package in R. This function returns the best ARIMA model according to either AIC, AICc or BIC value by conducting a search over possible model. Post that we create a custom function that will use auto.arima() to generate model order $$p,d,q$$ and then forecast it's RUL till it reaches 0.

```r
#Our custom function uses the auto.arima() function from forecast package
#Forecasts till 10000 iterations or when our RUL reaches 0, whichever is earlier
RUL_list_arima = lapply(final_list2, function(x){
  out = auto.arima(x, d = 1, max.p = 15, max.q = 15)
  p = out$arma[1]
  q = out$arma[2]
  for(i in 1:10000){
    out_for = sarima.for(x, i, p,1,q)
    if (out_for$pred[i] <= 0.0){
      RUL = i
      break
    }
  }
  return(RUL)
})
```
