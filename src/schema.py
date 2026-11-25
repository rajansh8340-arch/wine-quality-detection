from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=3.0, le=20.0, description="Fixed acidity (tartaric acid) in g/dm³", example=8.5)
    volatile_acidity: float = Field(..., ge=0.05, le=2.0, description="Volatile acidity (acetic acid) in g/dm³", example=0.28)
    citric_acid: float = Field(..., ge=0.0, le=1.5, description="Citric acid in g/dm³", example=0.56)
    residual_sugar: float = Field(..., ge=0.1, le=20.0, description="Residual sugar in g/dm³", example=1.8)
    chlorides: float = Field(..., ge=0.005, le=1.0, description="Sodium chloride in g/dm³", example=0.092)
    free_sulfur_dioxide: float = Field(..., ge=1.0, le=100.0, description="Free SO₂ in mg/dm³", example=35.0)
    total_sulfur_dioxide: float = Field(..., ge=5.0, le=350.0, description="Total SO₂ in mg/dm³", example=103.0)
    density: float = Field(..., ge=0.98, le=1.02, description="Density in g/cm³", example=0.9969)
    pH: float = Field(..., ge=2.5, le=4.5, description="pH level (0-14)", example=3.3)
    sulphates: float = Field(..., ge=0.2, le=2.5, description="Potassium sulphate in g/dm³", example=0.75)
    alcohol: float = Field(..., ge=7.0, le=16.0, description="Alcohol content (% vol)", example=12.8)

    def to_feature_dict(self) -> Dict[str, float]:
        """Convert pydantic model to dataframe column naming with spaces"""
        return {
            "fixed acidity": self.fixed_acidity,
            "volatile acidity": self.volatile_acidity,
            "citric acid": self.citric_acid,
            "residual sugar": self.residual_sugar,
            "chlorides": self.chlorides,
            "free sulfur dioxide": self.free_sulfur_dioxide,
            "total sulfur dioxide": self.total_sulfur_dioxide,
            "density": self.density,
            "pH": self.pH,
            "sulphates": self.sulphates,
            "alcohol": self.alcohol,
        }

class WineBatchInput(BaseModel):
    samples: List[WineFeatures]

class PredictionResult(BaseModel):
    is_good_quality: bool
    label: str
    confidence: float
    probability_good: float
    probability_bad: float
    key_drivers: Optional[Dict[str, float]] = None

class BatchPredictionResponse(BaseModel):
    total_samples: int
    good_quality_count: int
    bad_quality_count: int
    predictions: List[PredictionResult]

class ModelMetadataResponse(BaseModel):
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_samples: int
