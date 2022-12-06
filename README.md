# Classifying Fake Jobs
## Group Members: Zoe Lu and Cheslea Chen

### Project Description: This Project focuses on using a dataset from Kaggle with job posting information where we used the job description column to help us classify whether each job description is real or not. After cleaning the data, we used logistic regression support vector classification models to help us classify/ predict each job posting. 

#### Dataset Description
The "Real or Fake" (https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs) dataset we used was taken from Kaggle and has around 17,880 different job posting with 866 of them being fraduelnt postings. Various different types of columns are included including textual columns such as job posting description, location, title, etc. There were also numerical columns such as salary as well as binary columns such as if they would need to telecommute or answer screening questions. This dataset can be used in various ways such as training a model based on certain features to classify whether a job posting is fraudlent or not.

#### Packages Used: 
1. numpy (1.23.0)
2. pandas (1.5.2)
3. nltk (3.7)
4. re (3.11.0)
5. pickle (3.11.0)
6. sklearn (1.1.3)
7. wordcloud (1.8.2.2)

#### Demo File Description/ Instructions
The Demo File (demo_file) walks the audience through the process of some basic exploratory data analysis, cleaning textual data and later using it to build classifaction models to predict whether a job posting is fradulent or not. 

The first step in this file was running some basic exploratory data analysis with barcharts showing the counts of different columns in the data. For example, two barcharts were created with the first one showing the counts (y axis) of job postings that were fake or real (x axis). The second barchart shows the counts (y axis) of employment type (x axis). From here we can see that the data has majority real job postings and most of them were full-time jobs.

<img width="339" alt="Screen Shot 2022-12-05 at 5 48 03 PM" src="https://user-images.githubusercontent.com/97188472/205788544-e381d2b8-ed32-420e-ae5a-fccf4d28c62a.png">

<img width="338" alt="Screen Shot 2022-12-05 at 5 48 16 PM" src="https://user-images.githubusercontent.com/97188472/205788569-60632999-d2fe-48d6-b515-615ba3740c30.png">


Next, we cleaned the job description column since we will be using this column to help us classify whether or not a job posting is fake or not. To do this we .......

As we can see, before we processed the job description column, the descriptions have no punctuations, stopwords, etc and look like this:
<img width="979" alt="Screen Shot 2022-12-05 at 5 51 57 PM" src="https://user-images.githubusercontent.com/97188472/205789061-40a1b4f2-2d8f-421c-98f4-8d3e007ac49e.png">

After processing the job description columns, the descriptions no longer have stopwords, punctuations, etc and look like this:
<img width="987" alt="Screen Shot 2022-12-05 at 5 53 07 PM" src="https://user-images.githubusercontent.com/97188472/205789221-258f57f2-5c20-4934-9fd6-8438404c8aa1.png">



#### Scopes and Limitations
For this project we ran logistic regression, support vector machine, and decision trees classification models. An obstacle we ran into in the beginning was cleaning/ conducting the pre-processing step for the job description column. These obstacles included things like taking out puncuation and removing stop words. This process took us a while to figure out but we eventually figured it out by reading up on documentation online as well as looking at lecture notes. Some limitations of this project was that the data we had was heavily skewed where most of the job postings were not fradulent. Only about 800 of the job postings were actually fraudulent and this may have skewed/ hindered our model's classification results. We also did not run more data on the models to better train them.

In terms of accessibility, we feel that this project is quite acessible as the dataset is publicly available on Kaggle and there are a lot of resources online on how to build these classification models if needed. However, it maybe harder for people who do not have a background in machine learning/ classification to really undertsand what the code is doing if it's not documented well enough. 

When it comes to ethical implications, if this type of algorithm were to be used in real life, there exists the risk of a person applying to fraudlent job postings and getting scammed if the model classifies these job postings wrong. This could be extremely detrimental because a misclassification could lead to someone being scammed. 

To better improve this project, some future things we can do to improve are to balance the dataset so that the data is not so heavily skewed with way more entries for non-fraudulent data than fraudulent data. We can also try out more models and train it on more data to ensure that our model can predict/ classify well to prevent misclassification from occuriring at a high rate. With a lower misclassifiction rate, we hope this well decrease the chances of someone applying to a fake job and getting scammed as well. 

#### References and Acknowledgements
We would like to thank our professor Harlin and our TA Nikita for all the help and guidance this quarted!
