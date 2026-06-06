import sys
import os

from src.Pipeline.Predict_Pipeline import CustomData, PredictPipeline

from src.Pipeline.Predict_Pipeline import CustomData, PredictPipeline

def test_pipeline():
    print("Initializing test...")
    
    # 1. Create a sample instance of CustomData
    sample_data = CustomData(
        Temperature=32,
        RH=60,
        Ws=15,
        Rain=0.2,
        FFMC=78.1,
        DMC=24.5,
        DC=80.2,
        ISI=3.5,
        BUI=25.6,
        Classes="fire" 
    )

    # 2. Convert to DataFrame
    input_df = sample_data.get_data_as_df()
    print("Input DataFrame generated:")
    print(input_df.to_string(index=False))

    # 3. Initialize pipeline and predict
    print("\nInitializing PredictPipeline...")
    try:
        pipeline = PredictPipeline() 
        
        print("\nRunning prediction...")
        prediction = pipeline.predict(input_df)
        
        print(f"\n Success! The predicted FWI value is: {prediction[0]:.4f}")
        
    except Exception as e:
        print(f"\n Pipeline failed with error: {e}")

if __name__ == "__main__":
    test_pipeline()