# 🚗 Parking Space Classification

This project uses an Artificial Neural Network (ANN) to classify parking spaces as either vacant or occupied. The system processes images of parking areas and determines the status of each parking space.

## 📑 Table of Contents

- [📖 Description](#-description)
- [⚙️ Installation](#️-installation)
- [🚀 How to Run](#-how-to-run)
- [🤝 Contributions](#-contributions)
- [📝 Contact](#-contact)

## 📖 Description

Parking Space Classification is a machine learning project that leverages an Artificial Neural Network to identify whether parking spaces are vacant or occupied. The primary goal is to provide a reliable system for monitoring parking lots, which can be integrated into smart parking solutions.

## ⚙️ Installation

To set up this project on your local machine, follow these steps:

### Prerequisites

- 🖥️ **TensorFlow-GPU Installed**
  - Watch this video for setup instructions: [TensorFlow-GPU Setup Video](https://youtu.be/QUjtDIalh0k?feature=shared) (Not My Channel or Video)

### Steps

1. **Install Anaconda**: Follow the instructions in the video above to set up TensorFlow-GPU using Anaconda.

2. **Clone the repository**:

   ```sh
   git clone https://github.com/KrishnaGavali/parking_space_detection_V1
   cd parking_space_detection_V1
   ```

3. **Activate Conda Environment**:

   ```sh
   conda activate py310 #Name of your Anaconda Environment
   ```

4. **Install the required dependencies**:

   ```sh
   pip install -r module.txt
   ```

5. **Follow the rest of the instructions to run the code**
   [🚀 How to Run](#how-to-run)

---

## 🚀 How to Run

1. **Run the file `parking_space.py`**:

   - Check if the spaces are already selected.
   - Use LMB (Left Mouse Button) to select a space, RMB (Right Mouse Button) to deselect a space, and "q" to quit and save the selected spaces.

2. **Run the file `data_labelling.py`**:

   - Run the file.
   - Use "0" on the NumPad to label vacant spaces and "." for occupied spaces.(Be sure NumLock is on 😂)
   - Note: You will need to label/sort 1000+ images for training.

3. **Run the `model.ipynb`**:

   - Open and run all cells in the Jupyter notebook file.
   - **_Note: It will take time to train the model._**

4. **Run the file `mainV2.py`**:
   - Simply run the file.

---

## 🤝 Contributions

**Contributions are welcome! Please follow these steps to contribute:**

1. Fork the repository.
2. Create a new branch (`git checkout -b <branch_name>`).
3. Make your changes.
4. Commit your changes (`git commit -m "Add some feature"`).
5. Push your changes (`git push origin <branch_name>`).
6. Create a Pull Request 👍.

## 📝 Contact

If you have any questions or feedback, please reach out to:

- [Krishna Gavali on GitHub](https://github.com/KrishnaGavali)
- [Krishna Gavali on LinkedIn](https://www.linkedin.com/in/krishna-gavali/)
- [Krishna Gavali on Gmail](https://mail.google.com/mail/u/0/#inbox)

Feel free to get in touch!

---
