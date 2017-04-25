# Artificial Intelligence Engineer Nanodegree
## Probabilistic Models
## Project: Sign Language Recognition System


## Summary

To make the results a bit robust against randomness, all the feature sets (***Grounded Features***, ***Normalized Grounded Features***, ***Delta of Normalized Grounded Features***, ***Polar Features***, ***Combination Features*** for `features_ground`, `features_norm_grnd`, `features_delta_norm_grnd`, `features_polar` and `features_custom` respectively) were run for all the selection creteria (***Cross Validation Folds***, ***Bayesian Information Criterion***, ***Discriminative Information Criterion***) *three* times. The results for the same can be found in [results_set1.txt](./results_set1.txt), [results_set2.txt](./results_set2.txt) and [results_set3.txt](./results_set3.txt).

Following table summarizes the result for all the above combinations with average Word Error Rate (WER) for each of the combinations:

<img src="img/summary_table.png"></img>
<center><i>Table: Summary of Results</i></center>

**Best Feature Set:**

<img src="img/features_wer.png"></img>
<center><i>Fig: Performance of Feature Sets</i></center>

It can be seen from the above figure that ***Combination Features*** performs best despite the choice of selection creterion. This is because it feature space already contains the individual features group and thus benefits from the learning of other feature sets too.


**Selection Criteria:**

<img src="img/selector_wer.png"></img>
<center><i>Fig: Performance of Selection Criteria</i></center>

It can be seen from the above figure that ***DIC*** and ***BIC*** performs better than ***Cross Validation*** when it comes to model's parameter selection. This is as expected as the dataset is not big enough and dividing data in cross validation sets would lead to lesser number of samples for training. DIC makes the model stronger by competing with other word models and BIC generalizes the models by improving within-class statistics.

**Best Performing Combination:**
The best performing combination across the multiple run had been ***BIC with Combination Feature*** predicting 107 out of 178 words and thus giving an Word Error Rate (WER) of **39.88%**. On an average, the WER for BIC with combination feature was 0.42 and DIC with combination feature was 0.44. The best performing combination has the advantages of both the feature set and selection criteria as discussed above.


**How to imporve:**
The WER can be imporved by using Language Models. The basic idea is that each word has some probability of occurrence within the set, and some probability that it is adjacent to specific other words. We can use that additional information to make better choices. With this approach, sign language word recognition would use this probability together with the one obtained from the HMM to identify words. The current model is "0-gram" statistics that is it only consider probability of current word based on hmm models. "1-gram", "2-gram", and/or "3-gram" statistics can be used to enhance the performance of the recognition.


### Install

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html)
- [pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [jupyter](http://ipython.org/notebook.html)
- [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)

Notes: 
1. It is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python and load the environment included in the "Your conda env for AI ND" lesson.
2. The most recent development version of hmmlearn, 0.2.1, contains a bugfix related to the log function, which is used in this project.  In order to install this version of hmmearn, install it directly from its repo with the following command from within your activated Anaconda environment:
```sh
pip install git+https://github.com/hmmlearn/hmmlearn.git
```

### Code

A template notebook is provided as `asl_recognizer.ipynb`. The notebook is a combination tutorial and submission document.  Some of the codebase and some of your implementation will be external to the notebook. For submission, complete the **Submission** sections of each part.  This will include running your implementations in code notebook cells, answering analysis questions, and passing provided unit tests provided in the codebase and called out in the notebook. 

### Run

In a terminal or command window, navigate to the top-level project directory `AIND_recognizer/` (that contains this README) and run one of the following command:

`jupyter notebook asl_recognizer.ipynb`

This will open the Jupyter Notebook software and notebook in your browser. Follow the instructions in the notebook for completing the project.


### Additional Information
##### Provided Raw Data

The data in the `asl_recognizer/data/` directory was derived from 
the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php). 
The handpositions (`hand_condensed.csv`) are pulled directly from 
the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml). The three markers are:

*   0  speaker's left hand
*   1  speaker's right hand
*   2  speaker's nose
*   X and Y values of the video frame increase left to right and top to bottom.

Take a look at the sample [ASL recognizer video](http://www-i6.informatik.rwth-aachen.de/~dreuw/download/021.avi)
to see how the hand locations are tracked.

The videos are sentences with translations provided in the database.  
For purposes of this project, the sentences have been pre-segmented into words 
based on slow motion examination of the files.  
These segments are provided in the `train_words.csv` and `test_words.csv` files
in the form of start and end frames (inclusive).

The videos in the corpus include recordings from three different ASL speakers.
The mappings for the three speakers to video are included in the `speaker.csv` 
file.
