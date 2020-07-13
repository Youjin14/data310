{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Text classification with preprocessed text.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyPyZQX3MXpVCjeT4Cz3rXTr",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Youjin14/data310/blob/master/Text_classification_with_preprocessed_text.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "00utiBswVFVf",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "38b77c7f-dea0-4b6c-c2a2-a2653d570036"
      },
      "source": [
        "import tensorflow as tf\n",
        "\n",
        "from tensorflow import keras\n",
        "\n",
        "import tensorflow_datasets as tfds\n",
        "tfds.disable_progress_bar()\n",
        "\n",
        "import numpy as np\n",
        "\n",
        "print(tf.__version__)"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2.2.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IiNW-HwgVgnX",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 122
        },
        "outputId": "e2666c7c-5b2d-4b1c-b656-0bef81d951fc"
      },
      "source": [
        "(train_data, test_data), info = tfds.load(\n",
        "    # Use the version pre-encoded with an ~8k vocabulary.\n",
        "    'imdb_reviews/subwords8k', \n",
        "    # Return the train/test datasets as a tuple.\n",
        "    split = (tfds.Split.TRAIN, tfds.Split.TEST),\n",
        "    # Return (example, label) pairs from the dataset (instead of a dictionary).\n",
        "    as_supervised=True,\n",
        "    # Also return the `info` structure. \n",
        "    with_info=True)"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\u001b[1mDownloading and preparing dataset imdb_reviews/subwords8k/1.0.0 (download: 80.23 MiB, generated: Unknown size, total: 80.23 MiB) to /root/tensorflow_datasets/imdb_reviews/subwords8k/1.0.0...\u001b[0m\n",
            "Shuffling and writing examples to /root/tensorflow_datasets/imdb_reviews/subwords8k/1.0.0.incompleteHWHEZW/imdb_reviews-train.tfrecord\n",
            "Shuffling and writing examples to /root/tensorflow_datasets/imdb_reviews/subwords8k/1.0.0.incompleteHWHEZW/imdb_reviews-test.tfrecord\n",
            "Shuffling and writing examples to /root/tensorflow_datasets/imdb_reviews/subwords8k/1.0.0.incompleteHWHEZW/imdb_reviews-unsupervised.tfrecord\n",
            "\u001b[1mDataset imdb_reviews downloaded and prepared to /root/tensorflow_datasets/imdb_reviews/subwords8k/1.0.0. Subsequent calls will reuse this data.\u001b[0m\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BDvfLQADVmr2",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "53c5ee5e-1b8c-4a0b-8120-5ebb9b86c0a8"
      },
      "source": [
        "encoder = info.features['text'].encoder\n",
        "print ('Vocabulary size: {}'.format(encoder.vocab_size))"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Vocabulary size: 8185\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vdTFCpUrWFHZ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "ccc3883f-41b9-4c09-80e6-df7063ad1cf0"
      },
      "source": [
        "sample_string = 'Hello TensorFlow.'\n",
        "\n",
        "encoded_string = encoder.encode(sample_string)\n",
        "print ('Encoded string is {}'.format(encoded_string))\n",
        "\n",
        "original_string = encoder.decode(encoded_string)\n",
        "print ('The original string: \"{}\"'.format(original_string))\n",
        "\n",
        "assert original_string == sample_string"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Encoded string is [4025, 222, 6307, 2327, 4043, 2120, 7975]\n",
            "The original string: \"Hello TensorFlow.\"\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qHMZqjfNWFnT",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 136
        },
        "outputId": "abd61d6b-5637-424e-99ba-18d55442f692"
      },
      "source": [
        "for ts in encoded_string:\n",
        "  print ('{} ----> {}'.format(ts, encoder.decode([ts])))"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "4025 ----> Hell\n",
            "222 ----> o \n",
            "6307 ----> Ten\n",
            "2327 ----> sor\n",
            "4043 ----> Fl\n",
            "2120 ----> ow\n",
            "7975 ----> .\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vmLGaBWeWNaO",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "432d1a6f-bc12-4010-ffc3-cafbec454bc6"
      },
      "source": [
        "for train_example, train_label in train_data.take(1):\n",
        "  print('Encoded text:', train_example[:10].numpy())\n",
        "  print('Label:', train_label.numpy())"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Encoded text: [  62   18   41  604  927   65    3  644 7968   21]\n",
            "Label: 0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0A1zPYv_WP1D",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 103
        },
        "outputId": "3b8d3922-2996-4685-d1e4-188f3bdf8713"
      },
      "source": [
        "encoder.decode(train_example)"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic": {
              "type": "string"
            },
            "text/plain": [
              "\"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it.\""
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7qUJeKsVWSL_",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "BUFFER_SIZE = 1000\n",
        "\n",
        "train_batches = (\n",
        "    train_data\n",
        "    .shuffle(BUFFER_SIZE)\n",
        "    .padded_batch(32))\n",
        "\n",
        "test_batches = (\n",
        "    test_data\n",
        "    .padded_batch(32))"
      ],
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ADGeWVhyWUUK",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        },
        "outputId": "94eb9f2a-5e1e-4b35-b9ea-bf2f17960d1e"
      },
      "source": [
        "for example_batch, label_batch in train_batches.take(2):\n",
        "  print(\"Batch shape:\", example_batch.shape)\n",
        "  print(\"label shape:\", label_batch.shape)"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Batch shape: (32, 1357)\n",
            "label shape: (32,)\n",
            "Batch shape: (32, 1082)\n",
            "label shape: (32,)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pU5eUTRpWXYc",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 255
        },
        "outputId": "389ac178-a3b4-4576-a744-e01afa7d025d"
      },
      "source": [
        "model = keras.Sequential([\n",
        "  keras.layers.Embedding(encoder.vocab_size, 16),\n",
        "  keras.layers.GlobalAveragePooling1D(),\n",
        "  keras.layers.Dense(1)])\n",
        "\n",
        "model.summary()"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "embedding (Embedding)        (None, None, 16)          130960    \n",
            "_________________________________________________________________\n",
            "global_average_pooling1d (Gl (None, 16)                0         \n",
            "_________________________________________________________________\n",
            "dense (Dense)                (None, 1)                 17        \n",
            "=================================================================\n",
            "Total params: 130,977\n",
            "Trainable params: 130,977\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7pRIUo7KWZWK",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model.compile(optimizer='adam',\n",
        "              loss=tf.losses.BinaryCrossentropy(from_logits=True),\n",
        "              metrics=['accuracy'])"
      ],
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "X7kmkgiJWddR",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "f92af531-1998-4427-f652-fe3a975e4eab"
      },
      "source": [
        "history = model.fit(train_batches,\n",
        "                    epochs=30,\n",
        "                    validation_data=test_batches,\n",
        "                    validation_steps=30)"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.6818 - accuracy: 0.5002 - val_loss: 0.6653 - val_accuracy: 0.5052\n",
            "Epoch 2/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.6233 - accuracy: 0.5502 - val_loss: 0.5970 - val_accuracy: 0.6021\n",
            "Epoch 3/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.5447 - accuracy: 0.6594 - val_loss: 0.5349 - val_accuracy: 0.6927\n",
            "Epoch 4/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.4788 - accuracy: 0.7450 - val_loss: 0.4860 - val_accuracy: 0.7854\n",
            "Epoch 5/30\n",
            "782/782 [==============================] - 7s 10ms/step - loss: 0.4253 - accuracy: 0.8006 - val_loss: 0.4471 - val_accuracy: 0.8104\n",
            "Epoch 6/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.3836 - accuracy: 0.8305 - val_loss: 0.4193 - val_accuracy: 0.8417\n",
            "Epoch 7/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.3514 - accuracy: 0.8515 - val_loss: 0.3975 - val_accuracy: 0.8406\n",
            "Epoch 8/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.3251 - accuracy: 0.8666 - val_loss: 0.3824 - val_accuracy: 0.8490\n",
            "Epoch 9/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.3058 - accuracy: 0.8783 - val_loss: 0.3713 - val_accuracy: 0.8490\n",
            "Epoch 10/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2882 - accuracy: 0.8848 - val_loss: 0.3637 - val_accuracy: 0.8656\n",
            "Epoch 11/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2727 - accuracy: 0.8927 - val_loss: 0.3567 - val_accuracy: 0.8490\n",
            "Epoch 12/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2602 - accuracy: 0.8990 - val_loss: 0.3508 - val_accuracy: 0.8635\n",
            "Epoch 13/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2484 - accuracy: 0.9040 - val_loss: 0.3476 - val_accuracy: 0.8625\n",
            "Epoch 14/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2388 - accuracy: 0.9087 - val_loss: 0.3453 - val_accuracy: 0.8667\n",
            "Epoch 15/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2286 - accuracy: 0.9117 - val_loss: 0.3431 - val_accuracy: 0.8667\n",
            "Epoch 16/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2198 - accuracy: 0.9162 - val_loss: 0.3433 - val_accuracy: 0.8646\n",
            "Epoch 17/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2150 - accuracy: 0.9183 - val_loss: 0.3409 - val_accuracy: 0.8667\n",
            "Epoch 18/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2075 - accuracy: 0.9218 - val_loss: 0.3429 - val_accuracy: 0.8677\n",
            "Epoch 19/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.2009 - accuracy: 0.9242 - val_loss: 0.3418 - val_accuracy: 0.8729\n",
            "Epoch 20/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1935 - accuracy: 0.9262 - val_loss: 0.3434 - val_accuracy: 0.8677\n",
            "Epoch 21/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1887 - accuracy: 0.9301 - val_loss: 0.3466 - val_accuracy: 0.8677\n",
            "Epoch 22/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1810 - accuracy: 0.9319 - val_loss: 0.3497 - val_accuracy: 0.8656\n",
            "Epoch 23/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1787 - accuracy: 0.9351 - val_loss: 0.3572 - val_accuracy: 0.8594\n",
            "Epoch 24/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1728 - accuracy: 0.9359 - val_loss: 0.3517 - val_accuracy: 0.8646\n",
            "Epoch 25/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1680 - accuracy: 0.9388 - val_loss: 0.3548 - val_accuracy: 0.8729\n",
            "Epoch 26/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1650 - accuracy: 0.9395 - val_loss: 0.3565 - val_accuracy: 0.8687\n",
            "Epoch 27/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1601 - accuracy: 0.9428 - val_loss: 0.3601 - val_accuracy: 0.8708\n",
            "Epoch 28/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1555 - accuracy: 0.9430 - val_loss: 0.3642 - val_accuracy: 0.8719\n",
            "Epoch 29/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1514 - accuracy: 0.9458 - val_loss: 0.3675 - val_accuracy: 0.8594\n",
            "Epoch 30/30\n",
            "782/782 [==============================] - 7s 9ms/step - loss: 0.1483 - accuracy: 0.9468 - val_loss: 0.3703 - val_accuracy: 0.8656\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sy3REATUWe3T",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "1ee4986c-a0de-408b-edcf-99fb0d7169da"
      },
      "source": [
        "loss, accuracy = model.evaluate(test_batches)\n",
        "\n",
        "print(\"Loss: \", loss)\n",
        "print(\"Accuracy: \", accuracy)"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "782/782 [==============================] - 4s 5ms/step - loss: 0.3348 - accuracy: 0.8764\n",
            "Loss:  0.33478543162345886\n",
            "Accuracy:  0.8763599991798401\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "B07HACZ8Xbnl",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "a2e27ff6-9439-46e6-e278-9626fcef5895"
      },
      "source": [
        "history_dict = history.history\n",
        "history_dict.keys()"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "e_7RGiuCXsNO",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "outputId": "1d47fc1c-e137-408c-c63d-112ffefdcfcf"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "acc = history_dict['accuracy']\n",
        "val_acc = history_dict['val_accuracy']\n",
        "loss = history_dict['loss']\n",
        "val_loss = history_dict['val_loss']\n",
        "\n",
        "epochs = range(1, len(acc) + 1)\n",
        "\n",
        "# \"bo\" is for \"blue dot\"\n",
        "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
        "# b is for \"solid blue line\"\n",
        "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
        "plt.title('Training and validation loss')\n",
        "plt.xlabel('Epochs')\n",
        "plt.ylabel('Loss')\n",
        "plt.legend()\n",
        "\n",
        "plt.show()"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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KZ+Epri4mH/+85+ceuqpTJo0icLCQvr27cvUqVPjvr+goIDNmzcDMH36dPr06cMpp5xSPlQ1hHsEhgwZQv/+/bnwwgvZuXMnr7/+OvPmzeOWW25hwIABfPjhh4wfP56nnnoKgJdeeomBAwdy4okncuWVV/Lll1+Wf97UqVMZNGgQJ554IqtWrarx/NI9XHVOjD4aT4sWMHEi/PjH8P77obpIRGp3442wbFlyjzlgANx7b/Wvd+zYkaFDh/Lcc88xZswY5syZwyWXXIKZMX36dDp27Mi+ffs444wzWL58Of369Yt7nCVLljBnzhyWLVvG3r17GTRoEIMHDwbgggsu4JprrgHgX//1X/nNb37D97//fUaPHs25557LRRddVOlYu3fvZvz48bz00kv06dOHcePGcf/993PjjTcC0KlTJ5YuXcrMmTO55557eLiGRsl0D1edsyUCgKuvhry8yuMPJXtmJRFJjtjqodhqoSeffJJBgwYxcOBAVqxYUakap6rXXnuN888/n1atWtGuXTtGjx5d/to777zDqaeeyoknnsjs2bNZUcsIle+99x49e/akT/Qr8oorrmDBggXlr19wwQUADB48uHyguuosXLiQyy+/HIg/XPV9993H1q1badGiBUOGDOGRRx5h2rRpvP3227RNwng5OVsiAOjSBS68EB55BH72szAO0YQJFZNqlM2sBJrPQKRMTb/cU2nMmDHcdNNNLF26lJ07dzJ48GA++ugj7rnnHhYtWsShhx7K+PHjqx1+ujbjx49n7ty59O/fn0cffZT58+c3KN6yoawbMoz1lClTOOecc3j22WcZPnw4zz//fPlw1c888wzjx4/n5ptvZty4cQ2KNadLBBAajbdtgzlzap5ZSUTSq02bNowcOZIrr7yyvDSwfft2WrduTfv27dm4cSPPPfdcjcc47bTTmDt3Lrt27aK0tJQ///nP5a+VlpZy5JFHsmfPnvKhowHatm1LaWnpAcc69thjKS4uZvXq1QD87ne/4/TTT6/XuaV7uOqcLhEAnHIKnHACzJgRSgDxpGJmJRGpu7Fjx3L++eeXVxGVDdt83HHH0a1bN4YPH17j+wcNGsSll15K//79OfzwwxkyZEj5az/96U8ZNmwYnTt3ZtiwYeUX/8suu4xrrrmG++67r7yRGKBly5Y88sgjXHzxxezdu5chQ4Zw7bXX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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gqSCB3XcX28w",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "outputId": "e07cbd31-fd4a-4dbe-94d4-4bacdb168fb5"
      },
      "source": [
        "plt.clf()   # clear figure\n",
        "\n",
        "plt.plot(epochs, acc, 'bo', label='Training acc')\n",
        "plt.plot(epochs, val_acc, 'b', label='Validation acc')\n",
        "plt.title('Training and validation accuracy')\n",
        "plt.xlabel('Epochs')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.legend(loc='lower right')\n",
        "\n",
        "plt.show()"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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zJkzhw0bNnDwwQczYcKE3a69f+2113jzzTfZb7/9OPbYY3nuuecoLS3l4osvZt68efTv35+xY8cmjKlXr17885//pH379rz77ruMHTuWiooKnnrqKf7617/y0ksvUVhYyJo1awAoKyvj6quv5owzzmDz5s3U1NQ0/g+VRN4mgrlzQ5fTw4btnBc7Xb3mmvDH2q9fSAJqKBZJX79+oToo0fzGdtZZZ1EQ9SG/bt06LrjgAt59913MjG3btiX8zKmnnkq7du1o164dvXr14pNPPqFPnz67LDN06NAd8wYPHkxlZSWdOnXigAMO2HGd/tixY5k8efJu69+2bRuXX345CxcupKCggH/9618AzJo1iwsvvJDCwkIAunXrxoYNG1ixYgVnnHEGEG4Ka0p5WzU0dy6UlEDnzrvOLyuDyspwNVFlpZKASENNmhTOqONl6gy7Y8eOO17//Oc/Z+TIkbzxxhs8+eSTSa+pbxd381BBQQHV1dUNWiaZ2267jd69e7No0SIqKirqbczOprxMBF98AS+/rP6FRDKprAwmT4aiIjALz5MnZ/7gat26dey///4A/OlPf2r09R988MG8//77VFZWAjBt2rSkcey77760atWKhx56iO3btwPwla98hfvvv59NUR30mjVr6Ny5M3369GH69OkAbNmyZcf7TSEvE8FLL8HWrepfSCTTsnGGfdVVV/HTn/6UIUOGpHUEn6oOHTrwu9/9jtGjR1NSUkLnzp3p0qXLbstdeumlPPDAAwwaNIh33nlnx1nL6NGjOf300yktLWXw4MHceuutADz00EPccccdHHHEEQwfPpyPP/640WNPxmKt4LmitLTUKyoq9mgd118fHmvWQNeujRSYSB54++23OeSQQ7IdRtZt3LiRTp064e5cdtllDBgwgCuuuCLbYe2QaD+Z2SvunvD62bw8I5g3DwYPVhIQkYa55557GDx4MIcddhjr1q3j4osvznZIeyTvrhraujVcOhq7sUVEJF1XXHFFszoD2FN5d0awYEFoLFZDsYhIkHeJYN688Hz88dmNQ0Skuci7RDB3Lhx2GPToke1IRESah7xKBNXV8NxzqhYSEYmXV4ngtddCZ3NKBCK5aeTIkTz99NO7zLv99tuZMGFC0s+MGDGC2CXnX/va11i7du1uy0ycOHHH9fzJTJ8+nbfeemvH9HXXXcesWbPSCb/ZyqtEMHdueNaNZCK5aezYsUydOnWXeVOnTk3a8VttM2fOpGsDrxuvnQhuuOEGTj755Aatq7nJq8tH586Fgw6CffbJdiQiue+HPwxDvTamwYPh9tuTv3/mmWdy7bXXsnXrVtq2bUtlZSUfffQRxx9/PBMmTGDBggV88cUXnHnmmVx//fW7fb64uJiKigp69OjBpEmTeOCBB+jVqxd9+/alpKQECPcITJ48ma1bt3LggQfy0EMPsXDhQmbMmMHcuXO56aabeOKJJ7jxxhs57bTTOPPMMykvL+fKK6+kurqao446irvvvpt27dpRXFzMBRdcwJNPPsm2bdt47LHHGDhw4C4xVVZWcv755/P5558DcOedd+4YHOfmm2/m4YcfplWrVpxyyin86le/YunSpVxyySWsWrWKgoICHnvsMb70pS/t0e+eN2cE27fD/PmqFhLJZd26dWPo0KE89dRTQDgb+Pa3v42ZMWnSJCoqKli8eDFz585l8eLFSdfzyiuvMHXqVBYuXMjMmTNZsGDBjve++c1vsmDBAhYtWsQhhxzCvffey/Dhwzn99NO55ZZbWLhw4S4F7+bNmxk3bhzTpk3j9ddfp7q6mrvvvnvH+z169ODVV19lwoQJCaufYt1Vv/rqq0ybNm3HKGrx3VUvWrSIq666CgjdVV922WUsWrSI559/nn333XfPflTy6Izg9dfDYPWqFhJpHHUduWdSrHpozJgxTJ06lXvvvReARx99lMmTJ1NdXc3KlSt56623OOKIIxKuY/78+Zxxxhk7uoI+/fTTd7z3xhtvcO2117J27Vo2btzIV7/61TrjWbJkCf379+eggw4C4IILLuCuu+7ihz/8IRASC0BJSQl/+ctfdvt8c+iuOqNnBGY22syWmNlSM7s6wfvjzGyVmS2MHt/LRBxTpsBJJ4XXV1+95+Omikj2jBkzhvLycl599VU2bdpESUkJH3zwAbfeeivl5eUsXryYU089NWn30/UZN24cd955J6+//jq/+MUvGryemFhX1sm6sW4O3VVnLBGYWQFwF3AKcCgw1swOTbDoNHcfHD3+2NhxxMZN/eyzML1iRZhWMhDJTZ06dWLkyJFcdNFFOxqJ169fT8eOHenSpQuffPLJjqqjZE444QSmT5/OF198wYYNG3jyySd3vLdhwwb23Xdftm3bxpS4gqJz585s2LBh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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Zh6KqlHfYGbd",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "acc = history_dict['accuracy']\n",
        "val_acc = history_dict['val_accuracy']\n",
        "loss = history_dict['loss']\n",
        "val_loss = history_dict['val_loss']\n",
        "\n",
        "epochs = range(1, len(acc) + 1)\n",
        "\n",
        "# \"bo\" is for \"blue dot\"\n",
        "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
        "# b is for \"solid blue line\"\n",
        "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
        "plt.title('Training and validation loss')\n",
        "plt.xlabel('Epochs')\n",
        "plt.ylabel('Loss')\n",
        "plt.legend()\n",
        "\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
