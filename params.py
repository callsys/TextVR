






params = {
    'version' : 'fusion', # ocr   vision    offline   fusion
    
    # ./data/TextVR/TextVR_fusion.json     
    #  ./data/WebVid-2m/WebVid-2m.json
    
    
    #path TextVR
    'train_anns' : './data/TextVR/TextVR_train.json', 
    'val_anns' : './data/TextVR/TextVR_test_rand.json',
    'val_video_path' : "./data/TextVR",
    
    #path WebVid
    'WebVid_train_anns' : './data/WebVid-2m/WebVid-2m.json', 
    
    #path  MSRVTT
    'MSRVTT_train_anns' : './data/MSR-VTT/MSR-VTT_fusion.json', 
    'MSRVTT_val_anns' : './data/MSR-VTT/MSR-VTT_fusion.json',
    'MSRVTT_val_video_path' : "./data/MSR-VTT",
    
    
    'ocr_token_l' : 200,
}
