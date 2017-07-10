Softwares and Libraries used : 

1)Keras with tensorflow as backend
2)Anaconda distribution of Python 3.5.2

The zip file contains these file and folders : 

1) images_plots : Contains reference images and plots used in capstone project report.It also contains image of kaggle score submission for benchmark(knn classifier) and final model.

2)kaggle_csv_results : Contains 
   
	a)submit_augmented_final : results containing probability values found using final data augmentation model for 	getting kaggle 	score for 1531 testing images provided by kaggle.
  	
	b)submit_batch_drop : results containing probability values found using initial model with dropout and batch 		normalization  for getting kaggle score for 1531 testing images provided by kaggle.
  	
	c)submit_knn_raw_images:results containing probability values found using knn model(benchmark) for raw images 
	for getting kaggle score for 1531 testing images provided by kaggle.

	d)submit2_knn_extracted_features:results containing probability values found using knn model(benchmark) for 	extracted features for getting kaggle score for 1531 testing images provided by kaggle.

3)notebooks : contains : 

	a)CNN_augmentation.ipynb: final model notebook

	b)CNN_dropout_batch_normalize.ipynb:initial model notebook

	c)color_histogram_and_knn_classifier.ipynb: color histogram and knn(benchmark model) notebook

	d)data_shufle_and_validation.ipynb: data shuffling and creating validation sets notebook
	
	e)predicting_validation_data.ipynb: testing for validation set and plotting confusion matrix 

	f)preprocessing_images.ipynb: preprocessing images like resizing and centring

4) capstone_project.pdf : Final capstone report pdf

5) proposal.pdf : earlier submitted capstone proposal pdf
  	