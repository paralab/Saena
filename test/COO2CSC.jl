#address = "/home/abaris/Dropbox/Projects/julia/convert_COO2CSC/A_4x4.txt";
using LinearAlgebra
using SparseArrays

function COO2CSC(address)
  f = open(address)
  s = readlines(f);
  close(f);

  matrixInfo = split(s[1]);
  nz = parse(Int64, matrixInfo[3]);

  I = Array{Int64}(undef, nz);
  J = Array{Int64}(undef, nz);
  V = Array{Float64}(undef, nz);

  for i=1:nz
    temp = split(s[i+1]);
    I[i] = parse(Int64,temp[1]);
    J[i] = parse(Int64,temp[2]);
    V[i] = parse(Float64,temp[3]);
  end

  B = sparse(I, J, V);
  return B;
end
