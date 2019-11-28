include("COO2CSC.jl")
using Plots

# A = COO2CSC("../build/Helmholtz2D_modal_mat.mtx")
A = COO2CSC("./build/Helmholtz2D_modal_mat.mtx")
C = A * A
spy(C)
