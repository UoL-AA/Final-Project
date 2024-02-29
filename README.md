Since git has a file size limitation, the original train.csv file was split in two and was concatenated in the notebook using pandas.
Similarly, for the API to be able to load the pre-trained model, it needs the model.safetensors file and since the file is 0.5 GB, 
I decided to host it on https://drive.google.com/drive/u/0/folders/1qk7oLqE_XK9lQ8Ul8M0DqBlQeqKZ-Vgk , download and place inside the api/bert_model/ folder 
