# 📱 Algerian Forest Fire Prediction — Flutter Mobile UI

This is the cross-platform mobile frontend application for the **Algerian Forest Fire Prediction System**. Built natively using Flutter and Dart, this interface acts as the user-facing deployment tier of a complete, end-to-end MLOps pipeline.

---

## 🚀 Overview

The mobile application provides an intuitive, real-time interface for field researchers and environmental agencies to predict the likelihood of forest fires. Users can input specific meteorological parameters, which are securely serialized and processed by an isolated Machine Learning backend API.

### Key Architecture Roles
* **Frontend:** Flutter & Dart (Asynchronous UI & state management)
* **API Communication:** Handles secure RESTful network operations to exchange JSON payloads with the prediction server.
* **ML Pipeline Integration:** Seamlessly feeds user data into the trained machine learning model components managed in the root system.

---

## 🛠️ Tech Stack & Dependencies

* **Framework:** Flutter (Stable Channel)
* **Language:** Dart (With Sound Null Safety)
* **Target Platforms:** Android (Physical Device USB Debugging Verification) & iOS-ready

---

## 📁 Key Project Architecture

The essential codebase is modularly structured inside the `lib/` directory to separate presentation logic from network data layers:

```text
lib/
├── main.dart            # Application entry point and global configuration
├── models/              # Data models handling JSON serialization/deserialization
├── services/            # Isolated network services executing REST API operations
└── views/               # UI presentation layers, custom forms, and reusable components