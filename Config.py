

    

config=dict(
    # define model parameters
    start_token_id = 6,

    
    hidden_size = 128,
    num_heads = 8,
    num_layers = 6,
    input_length = 5,
    
    # seedfile is the file that contains the seed FRP
    seedfile="data/FRPseed/ICMPv6.txt",
    
    dataset='data/Population_Gungnir.csv',
    
    vocab_save_path='vocab_Gungnir.csv',
    
    routingprefix_file="data/routingprefix.txt",
    num_genarate_FRP=50,
    num_models=1,
    
    # define the path of the pyasn and lookuptable file
    pyasnfile="data/20250123.1600.dat",
    lookuptablefile="as_org_category.csv",
    
    # define the path of the model
    model_path='model/Gungnir',
    
    Prediction_path='Prediction/Prediction.csv',
    
    genarate_FRP_path="Prediction/PredictionFRP.txt",
    
    
)





