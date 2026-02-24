# ChandlerWobble
Mathematical Geoenergy treatment of Chandler wobble mechanism.

## Documentation

[<img width="680" height="642" alt="image" src="https://github.com/user-attachments/assets/b77d430f-3a17-40ad-882f-f32672562116" />](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/9781119434351.ch13)

- [`ChandlerWobbleForcing.pdf`](https://github.com/pukpr/ChandlerWobble/blob/main/ChandlerWobbleForcing.pdf) (2021 blog post: [https://geoenergymath.com/2021/01/07/chandler-wobble-forcing/](https://geoenergymath.com/2021/01/07/chandler-wobble-forcing/))
- [`dialog.pdf`](https://github.com/pukpr/ChandlerWobble/blob/main/dialog.pdf) (includes reference derivation and implementation notes)
-  The simulatiom generates a temporal behavior that reproduces spectral components observed in the Chandler wobble time time series, crucially certain sidebands that won't appear in the consensus model.
 

## Usage
Run the main simulation with the desired spectral estimator:

default FFT
```bash
python cw.py 
```
<img width="2404" height="1903" alt="image" src="https://github.com/user-attachments/assets/61856b65-7321-4d36-8a0d-06262f3b04fe" />

---

Maximum Entropy Model (MEM) order 400

```bash
python cw.py --spectral-estimator mem --ar-order 400
```
<img width="2404" height="1903" alt="image" src="https://github.com/user-attachments/assets/23bab957-245d-4f07-bfac-61d96e63d705" />

---
complex z-domain autoregressive (AR-z) order 100

```bash
python3 .\cw.py --spectral-estimator arz --ar-order 100
```
<img width="2404" height="1903" alt="image" src="https://github.com/user-attachments/assets/6e635201-641e-40fa-b57d-15d6d5b8ba4f" />
