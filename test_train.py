from train import train_model

def test_model_training():
    accuracy = train_model()

    assert accuracy >= 0.80
    assert accuracy <= 1.0