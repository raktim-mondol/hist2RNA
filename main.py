
       vae_model_single('./feature_extraction/VAE/'+aaenum+'/',x_train.shape[1],
       					x_train,x_test,intermediate_dim=1000,batch_size=20,latent_dim=50,epochs=100)
       ####### don't use the following lines when autoencoder requires fine tuning
       model = load_model('./feature_extraction/VAE/'+aaenum+'/vae_encoder'+'.h5')
       x_train = model.predict(x_train)
       print('X_Train Shape after VAE :', x_train.shape)
       x_test = model.predict(x_test)
       print('X_Test Shape after VAE :', x_test.shape)