function [M] = freq2mel(f)
% Frecuencias en escala normal a escala mel.
M = 1127*log(1 + f./700);
end

