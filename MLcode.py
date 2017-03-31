# Machine Learning pipeline
   ## ML library:  provide higher-level API built on top of DataFrames for constructing ML pipelines
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer,VectorAssembler, VectorIndexer 
from pyspark.ml.feature import  StringIndexer, OneHotEncoder
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier              # Decision tree
from pyspark.ml.classification import RandomForestClassifier              # Random Forest
from pyspark.ml.classification import MultilayerPerceptronClassifier      # Feedforward Artificial Neural network
from pyspark.ml.classification import GBTClassifier                       # Gradient-boosted tree 
from pyspark.ml.classification import NaiveBayes                          # Naive bayes
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegression                        # Regression problem
from pyspark.ml.classification import LogisticRegression                  # Logistic regression
from pyspark.ml.feature import PCA                                        # Principal Components Analysis 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

 #------------------------------------------------------------------------------------
    ## MLLIB library:  API built on top of RDD's
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.classification import NaiveBayesModel, NaiveBayes
from pyspark.mllib.classification import SVMModel, SVMWithSGD                  # Support Vector Machine
from pyspark.mllib.tree import DecisionTree, RandomForest, GradientBoostedTrees, GradientBoostedTreesModel 
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors

import time
from pyspark.sql.functions import avg
from pyspark.mllib.linalg import DenseVector, Vectors
from pyspark.sql.functions import udf
from pyspark.sql import Row
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit
from pyspark.sql.functions import avg
import pandas as pd


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql import DataFrame


conf = SparkConf().setAppName("App Name ")
sc = SparkContext(conf=conf)
#sqlContext = HiveContext(sc)
sqlContext = SQLContext(sc)
start = time.time()
starting = datetime.now() 
started = datetime.isoformat(starting)

print ('\nProcess Started on {}'.format(started))
print( "*"*100 ) 
##--------------------------------------------------------------------------------------------------------------------------



def MLPipe(df, Models, Grids, names, train_prop, df2):
    print( "\n**************************************************************" )   
    print( " Lets have some fun!!  ")
    print( "\n \n**************************************************************\n" ) 
    
    starting = datetime.now() 
    started = datetime.isoformat(starting)

    print ('\nProcess Started on {}'.format(started))
    print( "*"*100 ) 
    ## Assumption:: Your target varible is in the last column!! if not, adjust the following lines
    Cols = df.columns[:-1]
    Target = df.columns[-1]


    lw = 20
    print(" "*lw + '..Vectorizing Categorical Variables...\n')
    ## Assumption:: All the categorical and numerical variables will be used in the modeling process. If not, adjust the following 2 lines 
    categorical_features = [t[0] for t in df.dtypes if t[1] == 'string' and t[0] !=  df.columns[-1]]
    numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'double' or t[1] == 'bigint' and t[0] !=  df.columns[-1] ]
    
    ## pipeline stages 
    stages = []
    # Convert Target variable into label indices using the StringIndexer
    print (" "*lw +'..Converting Target variable into label indices...\n')
    label_stringIdx = StringIndexer(inputCol = Target, outputCol = "Label")
    stages += [label_stringIdx] 
    
    for col in categorical_features:
        stringIndexer = StringIndexer(inputCol = col, outputCol = col+'Index')
        encoder = OneHotEncoder(inputCol = col+'Index', outputCol = col+'classVec')
        stages += [stringIndexer, encoder]

  

    # Transform all features into a vector using VectorAssembler
    print (" "*lw +'..Transforming all features into a vector...\n')
    assemblerInputs = map(lambda c: c + "classVec", categorical_features) + numeric_features
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="Features")
    stages += [assembler]
       ## create a pipeline
    print('\nCreate ML Pipeline...')
    pipe = Pipeline(stages = stages)

    # Run the feature transformations.
    print ('\nTransform the entire dataset... ')

    pipedModel = pipe.fit(df)
    dataset = pipedModel.transform(df)

    # Keep relevant columns
    selectedcols = ["Label", "Features"] + df.columns
    dataset = dataset.select(selectedcols)
    ### Split data, randomly, into training and test sets. Stratified sampling is highly recommended,for skewed data
    print('\nSplit dataset into training and testing sets...')
    training, testing = dataset.randomSplit([train_prop, (1-train_prop)])
    ## Deployment 
    if df2 != '':
        stages.remove(stages[0])

        pipe1 = Pipeline(stages = stages)
        pipeNew = pipe1.fit(df)
        NewSet = pipeNew.transform(df2)
        # Keep relevant columns
        selectedcol = ["Features"] + df2.columns
        NewSets = NewSet.select(selectedcol)
    else:
        pass
   
    

    ##----------------------------------------------------------------------------------------------
    ## Run classification for each model
    for name, model, Grid in zip(names, Models, Grids):
        start = time.time()
        print( "\n Model: %s" % name)
    #     print('\nGrid: %s' % Grid)
    #     print('\nModel: %s' % model)
        print ('='*100)
        # hyper parameter tunning 
        # Search through sets of parameters for selection of best model
        paramGrid = Grid
        # Use precision score as the metric for evaluation
        evals = MulticlassClassificationEvaluator( labelCol = 'Label', predictionCol = 'prediction', metricName = 'precision')
             ##  perform k-fold cross-validation 
        CrossValidatedModel = (CrossValidator(estimator = model, estimatorParamMaps = Grid,  evaluator = evals, numFolds = 5))
        
        
        
        #  Fit the model
        CVmodel = CrossValidatedModel.fit(training)
        # Fetch best model
#         best = CVmodel.stages[2]
#         print ('\nBest Model:: {}'.format(best))
        # Fetch best model
        bestModel = CVmodel.bestModel
        print ('\nThe "best" Model, after cross validation:: {}'.format(bestModel))
        print('\n')
        predictions = CVmodel.transform(testing)
        ## Evaluation!!

        print('Evaluate the "best" model\n')
        evaluator = (MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction", metricName="precision"))
        accuracy = evaluator.evaluate(predictions)

        evaluator = MulticlassClassificationEvaluator(labelCol='Label', predictionCol='prediction', metricName='f1')  
        F1_score = evaluator.evaluate(predictions)

        evaluator = MulticlassClassificationEvaluator(labelCol='Label', predictionCol='prediction', metricName='recall')  
        recall = evaluator.evaluate(predictions)
        print ('Statistics::')
        print ('-'*50)       
#         if 'rawPrediction' not in predictions.columns:
#             pass
#         else:
#             evaluator2 = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
#             Auc = evaluator2.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
#             print ('Area Under the curve = {}'.format(Auc))
            

        print (" "*lw + 'Test Error = {}'.format(1.0 - accuracy))
        print (" "*lw + '-'*80)
        print (" "*lw + 'Accuracy = {}'.format(accuracy))
        print (" "*lw + '-'*80)
        print (" "*lw + 'Recall = {} '.format(recall))
        print (" "*lw + '-'*80)
        print (" "*lw + 'F1 score = {}'.format(F1_score))
        preds = predictions.select('Label', 'prediction')
        #predictions.columns
        #predictions.toPandas().head()
        ## confusion matrix
        print ('='*100)
        print ('Confusion Matrix::')
        print ('\n')
        preds.crosstab('Label', 'prediction').show()
        print ('='*100)
        print ('\n')
        ## deployment
        if df2 != '':
            print ('Deployment!')
            newPreds = bestModel.transform(NewSets)
            vector_udf = udf(lambda vector: float(vector[0]), DoubleType())
           
            if 'probability' not in newPreds.columns:                              ## some algorithms do not provide class probabilities 
                proba = newPreds.select(newPreds.id, newPreds.prediction)          ## here am selecting only variables 'id' and class. Select whatever fields that are important to you
            else:
                probs = (newPreds.select(newPreds.id, newPreds.probability))
                proba = probs.withColumn('probs1', vector_udf(probs.probability)).drop(probs.probability)
            
            ## save your results
            print('Saving ')
              ## as a table
            proba.write.mode('overwrite').saveAsTable('your_database.probs_{}'.format(name))
              ## or csv file 
            proba.toPandas().to_csv('path/to/your/file_{}.csv'.format(name))
              ## or can play with this
            updates = ('path/to/your/file_{}.csv'.format(name)) 
            open(updates,'w').close()
            Input = proba.rdd.collect()
            f = open(updates,'w')
            for i, row in enumerate(range(len(Input))): 
                f.write('%s,%s\n' %(Input[i][0], Input[i][1]))
            f.close() 
        else:
            pass
        stop = time.time()
        duration = stop -start
        print ('Time taken = {} seconds'.format(round(duration, 2)))
        #print ('='*100)
        print ("-"*100 ) 
        print( "*"*100 ) 
        print('\n\n\n')
## End 
 

##------------------------------------------------------------------------------------ 
## Provide the following. The machine wont do this for you :)

## Models
        
rf = RandomForestClassifier(labelCol="Label", featuresCol="Features")
GBT = GBTClassifier(labelCol="Label", featuresCol="Features")

Models = [rf, GBT]


## names
names = ['Random_Forest', 'Gradient_Boosting']


Grids = [(ParamGridBuilder()
              .addGrid(rf.maxDepth, [10, 30])  
              .addGrid(rf.maxBins, [40, 60])  
              .addGrid(rf.numTrees, [200, 600])
               .build()),
            
         (ParamGridBuilder()
             .addGrid(GBT.maxBins, [40, 60])
             .addGrid(GBT.maxDepth, [15, 30])
             .addGrid(GBT.maxIter, [8, 15, 20])
             .build())
            ]       


## Data
myData = sqlContext.table('your_database.your_table')
toScoreData = sqlContext.table('your_database.new_table')
##------------------------------------------------------------------------------------
## Run!

## Warning!!! The entire process may take several minutes to few hours depending on 
## the number of models, and the expansiveness of your grids          
%time MLPipe(myData, Models, Grids, names, 0.7, toScoreData)        
        
