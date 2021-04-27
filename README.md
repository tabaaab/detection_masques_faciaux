
![image](https://g.top4top.io/p_19435g1c11.jpg)

Realisé par : TABAA Abdelkader / M1 info ISIMA

COVID-19 : Face Mask Detector, développé un modèle de détection avec une précision de 99% en formation et en test. Détecter automatiquement si une personne porte un masque ou non dans les flux vidéo en temps réel

Table des matières
----------------
* Démo
* Vue d'ensemble
* Motivation
* Installation
* Caractéristiques
* Étapes/processus requis
* Structure du projet
* Résultat/Résumé
* Portée future du projet

Démo
----------
https://fr.linkedin.com/in/abdelkader-tabaa


Vue d'ensemble / Qu'est-ce que c'est ?
------------------------
* COVID-19 : Détecteur de masque facial avec OpenCV, Keras/TensorFlow et Deep Learning
* Détecter automatiquement si une personne porte un masque ou non dans des flux vidéo en temps réel.
* Notre objectif est d'entraîner un modèle d'apprentissage profond personnalisé pour détecter si une personne porte ou non un masque.

Motivation / Pourquoi/Raison ?
--------------------------------
Notre objectif est d'entraîner un modèle d'apprentissage profond personnalisé pour détecter si une personne porte ou non un masque.
* Augmentation continue et rapide de l'importance du virus covidien de porter un masque pendant les pandémies à ces moments-là a également été augmenté.
* L'utilisation universelle du masque peut réduire considérablement la transmission du virus dans les communautés
* Les masques et les protections faciales peuvent empêcher le porteur de transmettre le virus COVID-19 à d'autres personnes et lui conférer une certaine protection. 



Installation / Technique utilisée
------------------------------
Dataset : Real World Masked Face Dataset (RMFD)
AI/DL Techniques/Libaries : OpenCV, Keras/TensorFlow, MobileNetV2

Caractéristiques
-----------------


1-Le jeu de données  est composé de 1 376 images appartenant à deux classes :
    with_mask : 690 images
    sans_masque : 686 images

2-Les repères faciaux nous permettent de déduire automatiquement l'emplacement des structures faciales, notamment les yeux, les sourcils, le nez, la bouche et la mâchoire.

3-Une fois que nous savons où se trouve le visage dans l'image, nous pouvons extraire la région d'intérêt (ROI) du visage.



4-Et à partir de là, nous appliquons des repères faciaux, ce qui nous permet de localiser les yeux, le nez, la bouche, etc.



5- Ensuite, nous avons besoin d'une image d'un masque (avec un fond transparent) comme celle ci-dessous :



6- Ce masque sera automatiquement appliqué au visage en utilisant les repères du visage (à savoir les points le long du chi
n et du nez) pour calculer l'endroit où le masque sera placé.
Le masque est ensuite redimensionné et pivoté, pour le placer sur le visage 

7- Afin d'entraîner un détecteur de masque facial personnalisé, nous devons diviser notre projet en deux phases distinctes, chacune ayant ses propres sous-étapes respectives (comme le montre la figure 1 ci-dessus)::::: :

- Formation : Ici, nous nous concentrerons sur le chargement de notre jeu de données de détection de masque facial à partir du disque, l'entraînement d'un modèle (en utilisant Keras/TensorFlow) sur ce jeu de données, puis la sérialisation du détecteur de masque facial sur le disque.
- Déploiement : Une fois le détecteur de masque facial formé, nous pouvons passer au chargement du détecteur de masque, à la détection des visages, puis à la classification de chaque visage comme avec_masque ou sans_masque.




ÉTAPES/PROCESSUS REQ...
-----------------------
2.1. Extraction des données
2.2. Construction de la classe Dataset
2.3. Construction de notre modèle de détection de masques faciaux
2.4. Entraînement de notre modèle
2.5. Test de notre modèle sur des données réelles -> IMAGE/VIDEO
2.6. Résultats

Structure du projet::: :
-------------------------


Nécessite 3 scripts Python :: :
-----------------------------
** train_mask_detector.py : Accepte notre jeu de données d'entrée et ajuste MobileNetV2 sur celui-ci pour créer notre mask_detector.model. Un tracé de l'historique d'entraînement.png. contenant les courbes de précision/perte est également produit
* Detect_mask_video.py : En utilisant votre webcam, ce script applique la détection de masque de visage à chaque image du flux.
* Detect_mask_image.py : Effectue la détection de masque de visage dans des images statiques.

Dans les deux prochaines sections, nous allons entraîner notre détecteur de masque facial.
-----------------------------------------------------------
1 - Implémentation de notre script d'entraînement du détecteur de masque facial COVID-19 avec Keras et TensorFlow::::::: :

* Nous allons affiner l'architecture MobileNet V2, une architecture très efficace qui peut être appliquée à des appareils embarqués ayant une capacité de calcul limitée (ex. Raspberry Pi, Google Coral, NVIDIA Jetson Nano, etc.).

* Raison : Le déploiement de notre détecteur de masque facial sur des dispositifs embarqués pourrait réduire le coût de fabrication de tels systèmes de détection de masque facial, d'où notre choix d'utiliser cette architecture.

2- Entraînement du détecteur de masque facial COVID-19 avec Keras/TensorFlow



3 - Implémentation de notre détecteur de masque facial COVID-19 pour les images avec OpenCV

4 - Détection de masques faciaux COVID-19 dans des images avec OpenCV

5 - Implémentation de notre détecteur de masques faciaux COVID-19 dans des flux vidéo en temps réel avec OpenCV

6 - Détection de masques faciaux COVID-19 avec OpenCV dans des flux vidéo en temps réel 

ÉTAPES DE LA FORMATION DU MODÈLE
------------------------------
1 - De là, ouvrez un terminal, et exécutez la commande suivante :
$ python train_mask_detector.py --dataset dataset
[INFO] loading images...
[INFO] compiling model...
[INFO] training head...

Train for 34 steps, validate on 276 samples

Epoch 1/20
34/34 [==============================] - 30s 885ms/step - loss: 0.6431 - accuracy: 0.6676 -
val_loss: 0.3696 - val_accuracy: 0.8242

Epoch 2/20
34/34 [==============================] - 29s 853ms/step - loss: 0.3507 - accuracy: 0.8567 -
val_loss: 0.1964 - val_accuracy: 0.9375

Epoch 3/20
34/34 [==============================] - 27s 800ms/step - loss: 0.2792 - accuracy: 0.8820 -
val_loss: 0.1383 - val_accuracy: 0.9531

Epoch 4/20
34/34 [==============================] - 28s 814ms/step - loss: 0.2196 - accuracy: 0.9148 -
val_loss: 0.1306 - val_accuracy: 0.9492

Epoch 5/20
34/34 [==============================] - 27s 792ms/step - loss: 0.2006 - accuracy: 0.9213 -
val_loss: 0.0863 - val_accuracy: 0.9688

Epoch 16/20
34/34 [==============================] - 27s 801ms/step - loss: 0.0767 - accuracy: 0.9766 -
val_loss: 0.0291 - val_accuracy: 0.9922

Epoch 17/20
34/34 [==============================] - 27s 795ms/step - loss: 0.1042 - accuracy: 0.9616 -
val_loss: 0.0243 - val_accuracy: 1.0000

Epoch 18/20
34/34 [==============================] - 27s 796ms/step - loss: 0.0804 - accuracy: 0.9672 -
val_loss: 0.0244 - val_accuracy: 0.9961

Epoch 19/20
34/34 [==============================] - 27s 793ms/step - loss: 0.0836 - accuracy: 0.9710 -
val_loss: 0.0440 - val_accuracy: 0.9883

Epoch 20/20
34/34 [==============================] - 28s 838ms/step - loss: 0.0717 - accuracy: 0.9710 -
val_loss: 0.0270 - val_accuracy: 0.9922

[INFO] evaluating network...
 precision recall f1-score support
 with_mask 0.99 1.00 0.99 138
 without_mask 1.00 0.99 0.99 138
 accuracy 0.99 276
 macro avg 0.99 0.99 0.99 276
 weighted avg 0.99 0.99 0.99 276


2 - les courbes de précision/perte de formation démontrent → une grande précision et peu de signes de
d'ajustement excessif sur les données.

3 - a obtenu une précision de ~99% sur notre ensemble de test.

4 - En regardant la figure , nous pouvons voir qu'il y a peu de signes de sur-ajustement, la perte de validation étant inférieure à la perte d'apprentissage.
la perte de validation est inférieure à la perte d'apprentissage

5 - Au vu de ces résultats, nous avons bon espoir que notre modèle se généralisera bien aux
images en dehors de notre ensemble d'entraînement et de test.

RÉSUMÉ/RESULTAT
---------------
* Modèle de détection développé avec une précision de 97%, détectant automatiquement si une personne porte un masque ou non.
ou non dans des flux vidéo en temps réel

* Extraction du ROI du visage et des points de repère faciaux. Et appliqué au visage en utilisant le visage pour calculer.

* Utilisation de l'architecture MobileNetV2 hautement efficace et réglage fin de MobileNetV2 sur notre jeu de données masque/non-masque.
masque et sans masque et a obtenu un classificateur qui est 97% précis.

* Détermination du codage de l'étiquette de classe basé sur les probabilités associées à l'annotation de la couleur.

Portée future
------------
Peut être utilisé dans les caméras de vidéosurveillance pour capturer des personnes ou des groupes de personnes.

Peut être amélioré en fonction des besoins.
