Clustering
Preprocesss -> OpenFile -> Select the arff file
Cluster -> Pick Clusterer as SimpleKMeans or EM; 
        SimpleKMeans params: change numClusters and distanceMeasure
        EM params: change numClusters param

DimentionalityReduction: (Added the Student.Filters plugin from https://github.com/cgearhart/students-filters/blob/master/README.md)
    This plugin adds ICA to Weka
  Preprocess -> Open arff file -> Choose filter (weka->filters->unsupervised->attribute->IndependentComponents or Random Projection or PrincipalComponents)
-----------------
The classified outputs are obtained on Weka 3.7 Explorer and learning curves on Weka 3.7 Experimenter. Below are the steps to reproduce all results.

###Neural Network on Weka###

Preprocess -> Choose -> vsunkara6/winequalitywhite/train.arff
Classify -> Choose -> classifiers -> functions -> MultilayerPerceptron
Classify -> Test Options (Supplied test set) -> Open file -> vsunkara6/winequalitywhite/test.arff
Classify -> Start

Collect Classifier Output
--------------------
###Neural Network GUI###

Preprocess -> Choose -> vsunkara6/winequalitywhite/train_shortsample.arff
Classify -> Choose -> classifiers -> functions -> MultilayerPerceptron
Classify -> MultilayerPerceptron -> GUI:True; autoBuild:True; Hidden Layers:2,3,5;
                                    Learning Rate: 0.3; momentum: 0.2;
Classify -> Test Options (Supplied test set) -> Open file -> vsunkara6/winequalitywhite/test.arff
Classify -> Start
NN GUI - Start and Accept after 500 Epochs
------------------------------
###Learning Curve###
--------------------
- start the Experimenter
- select the configuration mode Advanced in the Setup panel
- choose as Destination an ARFF file
- open the options dialog of the CrossValidationResultProducer by left-clicking on the edit field
- open the options dialog for the splitEvaluator by left-clicking on the edit field
- choose the classifier that you want to analyze and setup it's parameters, in our case this is FilteredClassifier with one of the above classifiers as base classifier
- set the Generator properties to enabled
- choose as property percentage and click on Select: splitEvaluator -> classifier -> filter -> percentage
- add all the percentages that you want to test with removal, e.g. 90, 80, 70, 60, 50, 40, 30, 20, 10
- add the datasets you want to generate the learning curve for 
- go to the Run panel and start the experiment
- after the experiment has finished, select the Analyse panel and perform analysis on the results

###Data sets URLs###
Abalone - http://archive.ics.uci.edu/ml/datasets/Abalone 
Wine Quality - http://archive.ics.uci.edu/ml/datasets/Wine+Quality