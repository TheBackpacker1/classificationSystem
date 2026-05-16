# 🎓 Présentation Projet Master DABI : SmartFocus AI

*Ce document est structuré pour vous aider à créer vos diapositives de soutenance.*

---

## 🛝 Slide 1 : Titre & Introduction
**Titre suggéré :** SmartFocus AI : Classification Automatique de la Concentration par Vision Artificielle.
**Sous-titre :** Projet de Master 1 DABI - Suivi de l'engagement utilisateur en temps réel.
**Présenté par :** Mohamed Louati

---

## 🛝 Slide 2 : Contexte & Problématique
- **Contexte :** Avec l'essor du télétravail et de l'enseignement à distance, le suivi de l'engagement et de la concentration est devenu un défi majeur.
- **Problème :** Comment mesurer objectivement et en temps réel l'attention d'un utilisateur face à son écran, sans utiliser de matériel coûteux (uniquement une webcam standard) ?
- **Besoin :** Un système léger, respectueux de la vie privée (traitement local) et performant.

---

## 🛝 Slide 3 : La Solution Proposée
- **Approche :** Utilisation de l'Apprentissage Supervisé (Machine Learning) couplé à la Vision par Ordinateur.
- **Objectif :** Classifier l'état de l'utilisateur en 3 catégories distinctes :
  1. 🟢 **Concentré** : Posture droite, regard vers l'écran.
  2. 🟠 **Distrait** : Tête tournée, regard fuyant, utilisation du téléphone.
  3. 🔴 **Absent** : L'utilisateur n'est plus dans le champ de la caméra.

---

## 🛝 Slide 4 : La Méthodologie "Data-Centric"
*L'originalité du projet réside dans la création de notre propre vérité de terrain.*
- **Création d'un Dataset "Maison" :** Refus d'utiliser des bases de données génériques pour garantir une pertinence avec notre environnement réel.
- **Collecte :** Développement d'un script personnalisé (`capture_data.py`) permettant l'acquisition rapide d'images labellisées en temps réel.
- **Volume :** ~300 images (environ 100 par classe : Concentré, Distrait, Absent).
- **Évolution Multi-Utilisateurs :** Le système a été pensé pour pouvoir enrôler facilement de nouvelles personnes (Identification + Classification d'état).

---

## 🛝 Slide 5 : Le Pipeline Technique (Prétraitement)
Avant d'entraîner l'IA, les images subissent une standardisation stricte :
1. **Grayscale (Niveaux de gris) :** Pour rendre le modèle robuste aux changements de couleur (vêtements, lumières) et se concentrer sur la posture et les formes.
2. **Redimensionnement (64x64 pixels) :** Compression drastique pour permettre une inférence ultra-rapide en temps réel sans surcharger le processeur.
3. **Aplatissement & Normalisation :** Transformation de la matrice 2D en un vecteur mathématique 1D (valeurs entre 0 et 1) digeste pour les modèles classiques.

---

## 🛝 Slide 6 : L'Entraînement des Modèles
Comparaison de plusieurs algorithmes de Machine Learning classique pour trouver le meilleur compromis Vitesse/Précision :
- **Régression Logistique :** Pour sa simplicité et rapidité.
- **k-NN (k plus proches voisins) :** Pour sa classification basée sur la distance.
- **Random Forest (Forêt Aléatoire) :** L'algorithme retenu. Très robuste face au surapprentissage (overfitting) grâce à sa méthode d'ensemble.
- **Double Apprentissage :** Entraînement de deux modèles distincts : un pour l'Identité (qui est-ce ?) et un pour le Statut (que fait-il ?).

---

## 🛝 Slide 7 : Démonstration Temps Réel (Le Produit Fini)
*C'est ici que vous pourrez montrer une vidéo ou faire une démo en direct.*
- **Technologie embarquée :** OpenCV (Haar Cascades) pour la détection spatiale du visage.
- **Prédictions fluides :** Implémentation d'un algorithme de "Moyenne Glissante" (lissage sur les dernières images) pour éliminer le clignotement (flicker) et stabiliser la décision de l'IA.
- **Interface UI :** Affichage dynamique (Nom + État) directement traqué sur le visage de l'utilisateur, avec un rappel du statut global en haut de l'écran.

---

## 🛝 Slide 8 : Conclusion & Perspectives
- **Succès :** Le système est fonctionnel, léger, et répond parfaitement à la problématique initiale.
- **Limites actuelles :** Sensibilité aux conditions d'éclairage extrêmes (contre-jour).
- **Améliorations futures (V2) :** 
  - Passer au Deep Learning (CNN) si le volume de données augmente.
  - Intégrer l'analyse des "Facial Landmarks" (points clés du visage, ex: détection des clignements d'yeux via MediaPipe) pour affiner l'état de fatigue.
