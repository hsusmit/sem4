-- Simple ALU design with VHDL.

library IEEE;
use IEEE.std_logic_1164.all;


entity ALU is
    port(
        a,b: in std_logic_vector(1 downto 0);
        sel: in std_logic_vector(1 downto 0); -- select line
        res: out std_logic_vector(1 downto 0)
    );
end ALU;

architecture Behavioral of ALU is 
begin
    process(sel,a,b)
    begin
        case sel is
            when "00" =>
                res <= not b;
            when "01" =>
                res <= a and b;
            when "10" =>
                res <= a or b;
            when "11" =>
                res <= a xor b;    
            when others =>
                NULL;
        end case;
    end process;
end Behavioral;
