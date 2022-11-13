function [F] = mel2freq(m)
% Frecuencias en escala mel a escala normal.
F = (exp(m./1127)-1)*700;
end

