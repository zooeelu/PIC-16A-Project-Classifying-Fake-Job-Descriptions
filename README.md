# Classifying Fake Jobs
## Group Members: Zoe Lu and Cheslea Chen

### This Project focuses on using a dataset from Kaggle with job posting information where we used the job description column to help us classify whether each job description is real or not. After cleaning the data, we used logistic regression support vector classification models to help us classify/ predict each job posting. 

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

<img width="339" alt="Screen Shot 2022-12-05 at 5 48 03 PM" src="https://user-images.githubusercontent.com/97188472/205788544-e381d2b8-ed32-420e-ae5a-fccf4d28c62a.png"> <img width="338" alt="Screen Shot 2022-12-05 at 5 48 16 PM" src="https://user-images.githubusercontent.com/97188472/205788569-60632999-d2fe-48d6-b515-615ba3740c30.png">


Next, we cleaned the job description column since we will be using this column to help us classify whether or not a job posting is fake or not. To do this we created a class called Pre_Process(). This class will remove stop words, punctuations, etc. so that we're only left with important words. This class also throws an error if the elements in the job description column are not a string and does not have a length of at least 1.

As we can see, before we processed the job description column, the descriptions have no punctuations, stopwords, etc and look like this:

<img width="979" alt="Screen Shot 2022-12-05 at 5 51 57 PM" src="https://user-images.githubusercontent.com/97188472/205789061-40a1b4f2-2d8f-421c-98f4-8d3e007ac49e.png">

After processing the job description columns, the descriptions no longer have stopwords, punctuations, etc and look like this:

<img width="987" alt="Screen Shot 2022-12-05 at 5 53 07 PM" src="https://user-images.githubusercontent.com/97188472/205789221-258f57f2-5c20-4934-9fd6-8438404c8aa1.png">

We can also see in the word cloud below the most common types of words used. We did this by joining all the entries of the processed description column so that the WordCloud() function could count the frequencies of the words that appeared and generate the plot. 

<img width="686" alt="Screen Shot 2022-12-05 at 5 56 01 PM" src="https://user-images.githubusercontent.com/97188472/205789655-035c2e8c-899f-41d8-bf62-d27290b0d266.png">

After we processed the job descriptions, we then created a function called tfidf_train_test_split() that would calculate the importance of each word (TF-IDF) and also split our data into testing and training subsets. An error is rasied if the inputs X and Y are not one dimension and if X[0] is not an instance of string. Below is an image of some of the words and their TF-IDF Scores.

<img width="175" alt="Screen Shot 2022-12-05 at 6 18 34 PM" src="https://user-images.githubusercontent.com/97188472/205792480-1cc5e6f9-1f84-4cc6-9e20-dc8db97e9d6d.png">

Once our data was clean and split, we ran a Support Vector Machine model and used it with the data that included the processed descriptions and whether a job posting was fake or not. The results are shown below along with a confusion matrix as a visualization: 

<img width="347" alt="Screen Shot 2022-12-05 at 6 04 22 PM" src="https://user-images.githubusercontent.com/97188472/205790676-8cb56f39-0e08-4f19-99eb-91ef2a5cf594.png"> <img width="288" alt="Screen Shot 2022-12-05 at 6 04 44 PM" src="https://user-images.githubusercontent.com/97188472/205790723-4123ccc8-0b43-4ccf-b520-ae1a45406b12.png">

Next we ran a logistic regression model and used it with the data that included the processed descriptions and whether a job posting was fake or not. The results are shown below along with a confusion matrix as a visualization: 

<img width="342" alt="Screen Shot 2022-12-05 at 6 07 11 PM" src="https://user-images.githubusercontent.com/97188472/205791043-c977cc3d-9c81-4507-8fb9-8303b7b010ea.png"> <img width="288" alt="Screen Shot 2022-12-05 at 6 07 20 PM" src="https://user-images.githubusercontent.com/97188472/205791054-0aaabf1c-bd6f-4e65-a4d3-cdefae8024eb.png">

As a conclusion we compared the two different models to decide which model would be better to use to classify whether a job posting is fake or not. We chose a model based on the accuracy as well as the type of model and if one fits better for the type of data we have. 


#### Scopes and Limitations
For this project we ran logistic regression, support vector machine, and decision trees classification models. An obstacle we ran into in the beginning was cleaning/ conducting the pre-processing step for the job description column. These obstacles included things like taking out puncuation and removing stop words. This process took us a while to figure out but we eventually figured it out by reading up on documentation online as well as looking at lecture notes. Some limitations of this project was that the data we had was heavily skewed where most of the job postings were not fradulent. Only about 800 of the job postings were actually fraudulent and this may have skewed/ hindered our model's classification results. We also did not run more data on the models to better train them.

In terms of accessibility, we feel that this project is quite acessible as the dataset is publicly available on Kaggle and there are a lot of resources online on how to build these classification models if needed. However, it maybe harder for people who do not have a background in machine learning/ classification to really undertsand what the code is doing if it's not documented well enough. 

When it comes to ethical implications, if this type of algorithm were to be used in real life, there exists the risk of a person applying to fraudlent job postings and getting scammed if the model classifies these job postings wrong. This could be extremely detrimental because a misclassification could lead to someone being scammed. 

To better improve this project, some future things we can do to improve are to balance the dataset so that the data is not so heavily skewed with way more entries for non-fraudulent data than fraudulent data. We can also try out more models and train it on more data to ensure that our model can predict/ classify well to prevent misclassification from occuriring at a high rate. With a lower misclassifiction rate, we hope this well decrease the chances of someone applying to a fake job and getting scammed as well. 

#### References and Acknowledgements
We would like to thank our professor Harlin and our TA Nikita for all the help and guidance this quarted! We would also like to credit our STATS 101C professor Shrirong Xu's text cleaning We would also like the credit Kaggle for providing the data.
