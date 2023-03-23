-- Implement PIPO shift register using VHDL program.

library IEEE;
use IEEE.std_logic_1164.all;

entity pipo_reg is
    port (
        clk: in std_logic; 
        reset: in std_logic; 
        parallel_in: in std_logic_vector(3 downto 0); 
        parallel_out: out std_logic_vector(3 downto 0) 
    );
end pipo_reg;

architecture Behavioral of pipo_reg is
    signal shift_reg: std_logic_vector(3 downto 0);
begin
    shift_process: process(clk, reset)
    begin
        if reset = '1' then 
            shift_reg <= (others => '0');
        elsif rising_edge(clk) then 
            shift_reg <= parallel_in; 
        end if;
    end process shift_process;
    parallel_out <= shift_reg;
end Behavioral;