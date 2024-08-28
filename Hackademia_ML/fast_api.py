from fast_api import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

class Features(BaseModel):  
    features: conlist(float, min_items=4, max_items=4)
    
@app.post("/predict")
def predict(data: Features):
    try: 
        prediction = model.predict([data.features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))