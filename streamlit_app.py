{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "private_outputs": true,
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "#  Example Dataset\n",
        "data = pd.DataFrame({\n",
        "    'purring': [1,0,1,0,1,1,0,0,1,1],\n",
        "    'tail_up': [1,0,1,0,1,1,0,0,1,1],\n",
        "    'kneading': [1,0,1,0,0,1,0,0,1,1],\n",
        "    'meowing_often': [1,0,0,1,1,1,0,0,1,1],\n",
        "    'ears_forward': [1,0,1,0,1,1,0,0,1,1],\n",
        "    'eating_normally': [1,0,1,0,1,1,0,0,1,1],\n",
        "    'grooming': [1,0,1,0,1,1,0,0,1,1],\n",
        "    'hiding': [1,0,1,0,1,1,0,0,1,1],'asking to play': [1,0,1,0,1,1,0,0,1,1],\n",
        "    'happy': [1,0,1,0,1,1,0,0,1,1],\n",
        "})\n",
        "\n",
        "X = data.drop(columns='happy') #ONLY THE BEHAVIOURS EXCEPT THE RESULT FROM THE USERS!\n",
        "y = data['happy']\n",
        "#Train the model\n",
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=39)\n",
        "model=LogisticRegression() #Used Logistic as it predicts binary result well , 0 or 1 / happy or sad/ won or lost\n",
        "model.fit(X_train,y_train)\n",
        "\n",
        "# Predict\n",
        "y_pred = model.predict(X_test)\n",
        "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
        "\n",
        "your_cat = pd.DataFrame({\n",
        "    'purring': [1],\n",
        "    'tail_up': [1],\n",
        "    'kneading': [1],\n",
        "    'meowing_often': [0],\n",
        "    'ears_forward': [1],\n",
        "    'eating_normally': [1],\n",
        "    'grooming': [1],\n",
        "    'hiding': [1] , 'asking to play' :[0]\n",
        "})\n",
        "\n",
        "prediction = model.predict(your_cat)[0]\n",
        "if prediction==1:\n",
        "  print(\"Your Cat Is Happy😽😽\")\n",
        "else:\n",
        "  print(\"Your Cat Is not Happy!😿😿\")"
      ],
      "metadata": {
        "id": "WgGuLQ-9HPUr"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
