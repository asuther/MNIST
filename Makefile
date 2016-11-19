run:
	python ./src/test_script.py

clean-dataset:
	python ./src/clean_dataset.py $(datasetName)

generate-ellipse-features:
	python ./src/generate_ellipse_features.py $(datasetName)

test-pip:
	pip list