# Falcon-extension
Add some functionality (Sigmoid) for library Falcon.     
Use library [snwagh / falcon-public](https://github.com/snwagh/falcon-public).  
Note that the "Sigmoid" function S(x) here is different. Here S(x) = 2 * Sigmoid(x) - 1 (label y is 1 or -1), and then we apprximate it to x / 2 .    
Loss function is approximated to ln 2 - xy / 2 + x^2 / 8, and back propagation (gradient) is approximated to x / 4 - y / 2
