# Non-Target Boundary Attack Projects

This repository contains modified code from the original [Boundary Attack (ResNet)](https://github.com/greentfrapp/boundary-attack) to implement **Non-Target Boundary Attack**.  

The repository includes multiple experiments using different initial images for Non-Target Boundary Attack.

---

## File Descriptions

| File Name | Description |
|-----------|-------------|
| `noise_nontarget_csv.py` | Initial image is noise. Performs Non-Target Boundary Attack 2000 times. |
| `mean_nontarget_csv.py` | Initial image is a uniform image with the mean of the target image. Non-Target Boundary Attack 2000 times. |
| `red_nontarget_csv.py` | Initial image is a red uniform image. Non-Target Boundary Attack 2000 times. |
| `green_nontarget_csv.py` | Initial image is a green uniform image. Non-Target Boundary Attack 2000 times. |
| `blue_nontarget_csv.py` | Initial image is a blue uniform image. Non-Target Boundary Attack 2000 times. |
| `upgreen_mean_nontarget_csv.py` | Initial image is a uniform image with the mean of the target image and enhanced green channel. Non-Target Boundary Attack 2000 times. |
| `downgreen_mean_nontarget_csv.py` | Initial image is a uniform image with the mean of the target image and reduced green channel. Non-Target Boundary Attack 2000 times. |


---
## Installation

To run the scripts, you need the following Python packages. You can install them via pip:

```bash
pip install numpy matplotlib keras Pillow pandas opencv-python
```

## How to Run

Run with Python:

```bash
python noise_nontarget_csv.py
```


### Changing the Target Image

Modify the `target_sample` variable in the `boundary_attack()` function:

```python
target_sample = preprocess('images/original/seal.png')
```

---

## Major Modifications from Original Code

### `noise_nontarget_csv.py`
1. Changed from Target Boundary Attack → Non-Target Boundary Attack  
2. MSE is recorded in a CSV file  

### Other Files (`mean_nontarget_csv.py`, `upgreen_mean_nontarget_csv.py`, etc.)
1. Changed from Target Boundary Attack → Non-Target Boundary Attack  
2. MSE is recorded in a CSV file  
3. Added `get_sc_img(initial_sample)` function to generate uniform initial images  
4. Modified the "Move first step to the boundary" section

---

## Current Issues

- Due to randomness, the Boundary Attack may fail to converge in some cases.  
- This code is modified from the Target Boundary Attack, and may contain unresolved bugs.  
- Until resolved, it is recommended to consider using alternative Non-Target Boundary Attack implementations.

---

## Author

- Tan Yee Hang  
- Created: 2024/02/14  
- Contact: hang07020@gmail.com

