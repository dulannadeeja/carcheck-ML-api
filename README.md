# Carcheck: Revolutionizing the Sri Lankan Automotive Market through a Comprehensive and Transparent Online Vehicle Trading Platform

## [View full thesis report here](https://drive.google.com/file/d/1rKPBHpf-QTOJL44wN93ktj6cltljZkEm/view?usp=sharing) 

## [Server/Backend](https://github.com/dulannadeeja/Carcheck-server.git)
## [Client/Frontend](https://github.com/dulannadeeja/Carcheck-client.git)

# INTEGRATION OF MACHINE LEARNING:

Advanced machine learning models will be integrated into the platform to enable feature price prediction on vehicles. Developing a linear regression model for vehicle value prediction involves several steps, from data collection and preprocessing to training the model and evaluating its performance.

## 1.1 GATHERING DATA

Based on research, the developer identified key features that significantly influence a vehicle's value in the Sri Lankan market. These features include make, model, manufactured year, registered year, mileage, number of previous owners, exterior color, fuel type, condition, transmission, body type, and engine capacity. 

However, acquiring a dataset aligned with the Sri Lankan market proved challenging, as no existing dataset was available. To address this issue, the developer invested time in conducting market research on popular vehicle trading platforms such as ikman.lk and riyasewana.lk. From these platforms, the developer carefully extracted data to create a dataset suitable for training a price prediction model. This process was time-consuming, but ultimately, the developer obtained over 600 records to initiate the initial training process.

## 1.2 LOAD EXTRACTED DATASET TO THE SYSTEM

In this step, the developer read the extracted dataset from the CSV file and saved this data in the system. To accomplish this task, the developer utilized the Pandas library to read the data from the CSV file and then saved it to the model data collection.

## 1.3 SYNC WITH DATABASE

After loading the initial data, the system interacts with the Mongoose database to collect data from the listings collection posted on the website. This mechanism enables the machine learning model to update based on current market behaviors and collect a larger dataset for accurate price prediction. To achieve this, the machine learning model utilizes the Pymongo library to interact with the Mongoose database.

The data syncing logic implemented in the system keeps track of the IDs of records added to the model from the database. This mechanism simplifies the process of updating model data without overwriting all existing data from the database. Instead, it adds only the new records, ensuring efficient and seamless updates to the dataset.

## 1.4 PRE-PROCESSING DATA

Once the data is collected, the next step is preprocessing, which involves organizing the data in a format suitable for input into a linear regression model. This includes the following steps: 

Drop Null Values and Duplicates: Remove any null values and duplicate entries from the dataset to ensure data cleanliness and accuracy.

Map External Dataset Values: Map values from the external dataset to align with the system's data structure, ensuring consistency and compatibility.

Map Categorical Data: Convert categorical data, such as make, model, fuel type, transmission, body type, and exterior color, into numerical format using techniques like one-hot encoding. This process assigns a binary variable (0 or 1) to each category, enabling the model to interpret categorical data effectively.

Normalize Numerical Values: Normalize numerical variables, such as manufactured year, registered year, mileage, number of previous owners, and engine capacity, using techniques like scikit-learn's min-max scaler. Normalization brings these variables to a comparable scale, improving model efficiency and performance.

Shuffle Dataset: Shuffle the dataset to create more generalized chunks of data for training and testing
the model, enhancing its robustness and effectiveness.

By completing these preprocessing steps, the dataset is optimized and ready for input into the linear
regression model, facilitating accurate and reliable price predictions for vehicle listings.

## 1.5 TRAIN THE MODEL

After preprocessing, the next steps involve:

Splitting the Data: Divide the dataset into training and testing sets to independently validate the model's performance. 

Training the Model: Apply a linear regression algorithm to the training data. This entails fitting the model by estimating coefficients for each feature, minimizing the error between predicted and actual prices. Linear regression is a statistical method that models the relationship between a dependent variable (such as vehicle price) and one or more independent variables (features) using a linear function. For predicting vehicle value, the model interprets input features like make, model, year, mileage, etc., to estimate the vehicle's sale price accurately.

## 1.6 MODEL EVALUATION

The model uses the predict method on the test data to generate predictions for the dependent variable. developer calculates the R-squared value using the r2_score function from the sklearn.metrics module. The R-squared, or coefficient of determination, measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of goodness of fit and the percentage of the response variable variation that is explained by the linear model. 

The formula used is:
![MODEL EVALUATION](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhHLpjhrrXs-fcd0Ums3tg9w37FMZ5HbOjZN3yPqTO7TQOT7MYDWFuXmJ3CiAme4tv0DbKXq6UXJjApHuUTRSHumCscU4iiEiPWYq2BJ8D70iMuw16hy5laYjEjVREH2VdPYrbCM_sfR_6flabM7lpOZvz4aoGp1XsMNgLLwpfrxNKxw4OSMJeA-F2SdgRp/s16000/Screenshot%202024-05-07%20141413.png)
where the Sum of Squared Residuals is the sum of the squares of the model prediction errors, and the Total Sum of Squares is the sum of the squares of the difference between each dependent variable value and the mean of the dependent variable.

## 1.7 SERVER THE MODEL

For the price prediction server developer has chosen Python as the main technology. RESTful api layer has been layer using Fastapi freamwork. This server can serve response of predicted value when input features are provided. This machine learning model is implemented a way that admins of the system can interact with the lifecycle of the model training process through the API endpoints.

## 1.8 INTEGRATION INTO THE APPLICATION

For the price prediction server, the developer chose Python as the primary technology. The RESTful API layer was built using the FastAPI framework.

This server can efficiently serve responses for predicted values when input features are provided. Additionally, the machine learning model was implemented in a manner that allows administrators of the system to interact with the lifecycle of the model training process through API endpoints. This enables administrators to manage model training, evaluation, and updates seamlessly within the system.

![MODEL EVALUATION](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjHrTHnhvG39QhdPWrPY0ANbXCyvzA6o8pNU5N8n1vgvlvTc0FbF23DDvLhUjtJ7u6Ki4S7c-F_JmtDT4Q6_-MJ8QwfhFvrxHcMpUMUMbMVPtnBtGfDG5wu3VKBga3hKqe1AC-RWvKBLPoGxovjxtLe68f1RQkeKOGwH26kZHYo4H79JWK3sTYqJ3IF3dSJ/s1652/system%20settings.png)

![Create a New Listing Feature](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjZsE5jbcJfjSMgm1FHkmVgYiYYnFIUdpq6mj6JGZGuln5XF_7eT0ujmIPn1sVGmXLw2z3fVzT0-4MmNJirD90NZezcZTJr0SFWzsIvB9jFQJ8-Qzp_qrthEuj6Ls_ezO7JBIDfStF1g7YNAJzZYYvdgnmHJ1b1BgItn8mxY6tVQWdml1OSpSc5IQ5Sl2Or/s16000/Screenshot%202024-05-21%20161019.png) 

# DEPLOYMENT AND MONITORING:

The fully developed platform will be deployed and monitored to ensure smooth operation and functionality. The deployment will involve setting up the environment on AWS EC2 instances for different components—Python model, Node.js REST API, and the React client—to ensure continuous uptime and public accessibility. 



