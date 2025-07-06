function QPSK_symbol = QPSK_Modulator(dataSym)

N = size(dataSym, 1) * size(dataSym, 2);

QPSK_symbol = zeros(N, 1);

QPSK_symbol(( dataSym == 0)) = (1 + 1j)/sqrt(2);
QPSK_symbol(( dataSym == 1)) = (-1 + 1j)/sqrt(2);
QPSK_symbol(( dataSym == 3)) = (-1 - 1j)/sqrt(2);
QPSK_symbol(( dataSym == 2)) = (1 - 1j)/sqrt(2);% 2020.06.29

