-- Design arithmetic unit for ALU simple unit with VHDL.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_unsigned.all;
use IEEE.std_logic_arith.all;

entity arithmetic_unit is
    port(
        a,b: in std_logic_vector(3 downto 0);
        op: in std_logic_vector(2 downto 0);
        f: out std_logic_vector(3 downto 0)
    );
end arithmetic_unit;

architecture Behavioral of arithmetic_unit is 
begin
    process(op,a,b)
    variable temp: std_logic_vector(3 downto 0);
    begin
        case op is
            when "000" =>
                temp := a+b;
            when "001" =>
                temp := a+b+1; 
            when "010" =>
                temp := a+(not b)+1; 
            when "011" =>
                temp := a+(not b); 
            when "100" =>
                temp := a+1; 
            when "101" =>
                temp := a-1; 
            when "110" =>
                temp := a; 
            when "111" =>
                temp := b; 
            when others =>
                NULL;
        end case;
        f <= temp;
    end process;
end Behavioral;
       