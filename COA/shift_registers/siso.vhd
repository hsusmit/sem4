-- Implement SISO shift register using VHDL program.

library IEEE;
use IEEE.std_logic_1164.all;

entity siso_reg is
    port (
        clk: in std_logic; 
        reset: in std_logic; 
        serial_in: in std_logic; 
        serial_out: out std_logic 
    );
end siso_reg;

architecture Behavioral of siso_reg is
    signal shift_reg: std_logic_vector(3 downto 0);
begin
    shift_process: process(clk, reset)
    begin
        if reset = '1' then 
            shift_reg <= (others => '0');
        elsif rising_edge(clk) then 
            shift_reg <= serial_in & shift_reg(3 downto 1); 
        end if;
    end process shift_process;
    serial_out <= shift_reg(0);
end Behavioral;