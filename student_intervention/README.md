
# Building a Student Intervention System

## Learning Type

Supervised Learning

## Project Brief

As education has grown to rely more and more on technology, more and more data is available for examination and prediction. Logs of student activities, grades, interactions with teachers and fellow students, and more are now captured through learning management systems like Canvas and Edmodo and available in real time. This is especially true for online classrooms, which are becoming more and more popular even at the middle and high school levels.

Within all levels of education, there exists a push to help increase the likelihood of student success without watering down the education or engaging in behaviors that raise the likelihood of passing metrics without improving the actual underlying learning. Graduation rates are often the criteria of choice for this, and educators and administrators are after new ways to predict success and failure early enough to stage effective interventions, as well as to identify the effectiveness of different interventions.

Toward that end, your goal as a software engineer hired by the local school district is to model the factors that predict how likely a student is to pass their high school final exam. The school district has a goal to reach a 95% graduation rate by the end of the decade by identifying students who need intervention before they drop out of school. You being a clever engineer decide to implement a student intervention system using concepts you learned from supervised machine learning. Instead of buying expensive servers or implementing new data models from the ground up, you reach out to a 3rd party company who can provide you the necessary software libraries and servers to run your software.

However, with limited resources and budgets, the board of supervisors wants you to find the most effective model with the least amount of computation costs (you pay the company by the memory and CPU time you use on their servers). In order to build the intervention software, you first will need to analyze the dataset on students’ performance. Your goal is to choose and develop a model that will predict the likelihood that a given student will pass, thus helping diagnose whether or not an intervention is necessary. Your model must be developed based on a subset of the data that we provide to you, and it will be tested against a subset of the data that is kept hidden from the learning algorithm, in order to test the model’s effectiveness on data outside the training set.

Your model will be evaluated on three factors:

* Its F1 score, summarizing the number of correct positives and correct negatives out of all possible cases. In other words, how well does the model differentiate likely passes from failures?
* The size of the training set, preferring smaller training sets over larger ones. That is, how much data does the model need to make a reasonable prediction?
* The computation resources to make a reliable prediction. How much time and memory is required to correctly identify students that need intervention?

## Dataset Description

Attributes for student-data.csv:

- school - student's school (binary: "GP" or "MS")
- sex - student's sex (binary: "F" - female or "M" - male)
- age - student's age (numeric: from 15 to 22)
- address - student's home address type (binary: "U" - urban or "R" - rural)
- famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
- Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
- Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
- Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
- Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
- guardian - student's guardian (nominal: "mother", "father" or "other")
- traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
- studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
- failures - number of past class failures (numeric: n if 1<=n<3, else 4)
- schoolsup - extra educational support (binary: yes or no)
- famsup - family educational support (binary: yes or no)
- paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
- activities - extra-curricular activities (binary: yes or no)
- nursery - attended nursery school (binary: yes or no)
- higher - wants to take higher education (binary: yes or no)
- internet - Internet access at home (binary: yes or no)
- romantic - with a romantic relationship (binary: yes or no)
- famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
- freetime - free time after school (numeric: from 1 - very low to 5 - very high)
- goout - going out with friends (numeric: from 1 - very low to 5 - very high)
- Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
- Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
- health - current health status (numeric: from 1 - very bad to 5 - very good)
- absences - number of school absences (numeric: from 0 to 93)
- passed - did the student pass the final exam (binary: yes or no)

## 1. Classification vs Regression

In this we are essentially building a system that identifies student in need of early intervention to adress the potential future difficulties that may probably come to students in academic area.Taking this in mind we come up with some parameters(features) of previous student's data like their school,age,activities,failures,absences etc. that we will be feeding in an algorithm to train itself to be able to predict if the student passed or not in the final exam,i.e. a categorical yes/no output after taking into account all the features and data feeded into the algorithm.

Essentially what separates classification and regression is the way the output variable is produced.In classification we are "classifying" the data in some predefined classes or groups(mostly in binary,but it can be more than two depending on the task in hand.).However,in regression we deal with continuos data like predicting house prices,or predicting marks of a student in glass(while predicting grades would be a classification problem).Although the boundary between classification and regression is fuzzy,one can think the difference between continuos or categorical data by taking up examples like is age continuos or categorical(continuos),are grades of student continuos or categorical(categorical)

As the student intervention project deals with categorical output(Binary to be precise),it is a classification problem.

## 2. Exploring the Data


    Total number of students: 395
    Number of students who passed: 265
    Number of students who failed: 130
    Number of features: 30
    Graduation rate of the class: 67.09%
    

## 3. Preparing the Data
In this section, we will prepare the data for modeling, training and testing.

### Identify feature and target columns
It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.


    Feature column(s):-
    ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    Target column: passed
    
    Feature values:-
      school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
    0     GP   F   18       U     GT3       A     4     4  at_home   teacher   
    1     GP   F   17       U     GT3       T     1     1  at_home     other   
    2     GP   F   15       U     LE3       T     1     1  at_home     other   
    3     GP   F   15       U     GT3       T     4     2   health  services   
    4     GP   F   16       U     GT3       T     3     3    other     other   
    
        ...    higher internet  romantic  famrel  freetime goout Dalc Walc health  \
    0   ...       yes       no        no       4         3     4    1    1      3   
    1   ...       yes      yes        no       5         3     3    1    1      3   
    2   ...       yes      yes        no       4         3     2    2    3      3   
    3   ...       yes      yes       yes       3         2     2    1    1      5   
    4   ...       yes       no        no       4         3     2    1    2      5   
    
      absences  
    0        6  
    1        4  
    2       10  
    3        2  
    4        4  
    
    [5 rows x 30 columns]
    

## 4. Training and Evaluating Models
Choose 3 supervised learning models that are available in scikit-learn, and appropriate for this problem. For each model:

- What are the general applications of this model? What are its strengths and weaknesses?
- Given what you know about the data so far, why did you choose this model to apply?
- Fit this model to the training data, try to predict labels (for both training and test sets), and measure the F<sub>1</sub> score. Repeat this process with different training set sizes (100, 200, 300), keeping test set constant.

Produce a table showing training time, prediction time, F<sub>1</sub> score on training set and F<sub>1</sub> score on test set, for each training set size.

Note: You need to produce 3 such tables - one for each model.

### Random Forest

## Description:

* A Random Forest has a space complexity of O(√d n log n) with d is the number of features and n the number of elements in the dataset, under the assumption that a reasonably symmetric tree is built.
* The training complexity is given as O(M √d n log n), where M denotes the number of trees. The training complexity is greater than the prediction by a factor of M, such that training time would be ten times (the numner of default trees in the algorithm) that of prediction time.
* Random Forest learners have been implemented in numerous data mining applications in fields from agriculture, genetics, medicine, physics to text processing - even the Xbox Kinect.

Pros:

* Random forest strength is that it can scale well as runtimes are quite fast, and they are able to deal with unbalanced and missing data.

Cons:

* Random Forest weaknesses are that when used for regression they cannot predict beyond the range in the training data, and that they may over-fit data sets that are particularly noisy.

Reasons for Selection:

* This algorithm was chosen as it is an ensemble of decision tree classifiers, which might suit the dataset well given that majority of the features appear to be mutually exclusive from each other.

| Training Size         | 100            | 200            | 300            |
|-----------------------|----------------|----------------|----------------|
| Prediction Time       | 0.02           | 0.02           | 0.03           |
| F1 Score Training Set | 1.0            | 1.0            | 1.0            |
| F1 Score Test Set     | 0.754098360656 | 0.758620689655 | 0.779661016949 |


### AdaBoost Classifier

## Description:

* AdaBoost, short for "Adaptive Boosting", is a machine learning meta-algorithm  is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

Pros:
* Unlike other powerful classifiers, such as SVM, AdaBoost can achieve similar classification results with much less tweaking of parameters or settings (unless of course you choose to use SVM with AdaBoost).The user only needs to choose: (1) which weak classifier might work best to solve their given classification problem; (2) the number of boosting rounds that should be used during the training phase. The GRT enables a user to add several weak classifiers to the family of weak classifiers that should be used at each round of boosting. The AdaBoost algorithm will select the weak classifier that works best at that round of boosting.

Cons: 
* AdaBoost can be sensitive to noisy data and outliers. In some problems, however, it can be less susceptible to the overfitting problem than most learning algorithms. The GRT AdaBoost algorithm does not currently support null rejection, although this will be added at some point in the near future.

Reasons for Selection:
* With Adaboosting we can try out different weak learners and as it is a non parametric model we have not to make any assumptions about the data,much like SVM they can capture very complex decision boundries.

| Training Size         | 100            | 200            | 300            |
|-----------------------|----------------|----------------|----------------|
| Prediction Time       | 0.01           | 0.01           | 0.01           |
| F1 Score Training Set | 0.855263157895 | 0.821548821549 | 0.810572687225 |
| F1 Score Test Set     | 0.813559322034 | 0.8            | 0.825396825397 |


### Multinomial Naive Bayes

## Description:

* In machine learning, Multinomial Naive Bayes is essentially a generalised version of Naive Bayes algorithm,it is generally used for multi class data and is considered as the fastest and lightest form of algorithm,its complexity being linear in time.

Pros :
* Simplicity,powerful and efficient learning with less computational resources and time.

Cons : 
* Some of it's weakness include its poor performance if independence assumptions do not hold and has difficulty with zero-frequency values.

Reasons for selection : 
* To check how far a simple and fast algorithm can go.

| Training Size         | 100            | 200            | 300            |
|-----------------------|----------------|----------------|----------------|
| Prediction Time       | 0.00           | 0.00           | 0.00           |
| F1 Score Training Set | 0.807453416149 | 0.798634812287 | 0.798206278027 |
| F1 Score Test Set     | 0.825396825397 | 0.852459016393 | 0.833333333333 |


## 5. Choosing the Best Model

Based on the above statistics its clear that the algorithm that gives best results on this data is AdaBoosting Classifier.While choosing a algorithm i used randomly as n_estimators parameter to be 3(by default its 50,which is very large for this given data,as boosting in most cases converges to atmost 10 n_estimators).We can use this parameter to fine tune the algorithm and use gridSearchCV algorithm for this task.I chose the n_estimators from 1 to 5 and ran the code looping over 100 times and checking each time the best estimator.

The results are described as below :

```
F1 Scores count    100.000000
mean       0.808454
std        0.035327
min        0.714286
25%        0.786593
50%        0.813559
75%        0.833333
max        0.900000
dtype: float64
n_estimators count    100.000000
mean       1.910000
std        1.256056
min        1.000000
25%        1.000000
50%        1.000000
75%        3.000000
max        5.000000
dtype: float64
```
