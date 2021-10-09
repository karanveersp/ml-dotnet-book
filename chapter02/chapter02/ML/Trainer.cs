using chapter02.ML.Base;
using chapter02.ML.Objects;
using Microsoft.ML;
using System;
using System.IO;

namespace chapter02.ML
{
    public class Trainer : BaseML
    {
        public void Train(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName})");
                return;
            }

            // Load csv data into memory as class instances (like a generator)
            var trainingDataView = MlContext.Data.LoadFromTextFile<RestaurantFeedback>(trainingFileName);

            // Splitting training data from test data.
            var dataSplit = MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            // Creating the pipeline.
            // Here we are mapping data to Features, which will be used by the trainer algorithm.
            var dataProcessPipeline = MlContext.Transforms.Text.FeaturizeText(
                inputColumnName: nameof(RestaurantFeedback.Text),
                outputColumnName: "Features");

            // Instantiating an SDCA trainer. We tell it the label column, and
            // the feature column.
            var sdcaRegressionTrainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                featureColumnName: "Features",
                labelColumnName: nameof(RestaurantFeedback.Label));

            // Complete the pipeline by appending the trainer.
            var trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);

            // Train the model with the dataset, loading the data, and running the pipeline.
            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            // Save newly created model to the file, while matching the training set's schema.
            MlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);

            // Transform the newly created model with the test set created earlier.
            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            var modelMetrics = MlContext.BinaryClassification.Evaluate(
                data: testSetTransform,
                labelColumnName: nameof(RestaurantFeedback.Label),
                scoreColumnName: nameof(RestaurantPrediction.Score));

            Console.WriteLine(
                $"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}{Environment.NewLine}" +
                $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}{Environment.NewLine}" +
                $"Accuracy: {modelMetrics.Accuracy:P2}{Environment.NewLine}" +
                $"F1Score: {modelMetrics.F1Score:P2}{Environment.NewLine}" +
                $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
                $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}");
        }
    }
}