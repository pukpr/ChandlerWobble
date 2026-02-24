with Text_IO;
with Ada.Numerics.Long_Elementary_Functions;

procedure CW is
   use Ada.Numerics, Ada.Numerics.Long_Elementary_Functions;
   
   subtype Ordinal is Integer range 1..10_000;
   
   type TS is array(Ordinal) of Long_Float;
   subtype Float_Array is TS;
   
   Dates, Values : TS;
   
   Year : constant := 365.242;
   Draconic : constant := 27.2122;
   Anomalistic : constant := 27.5545;
   Tropical : constant := 27.3216;
   DA : constant := 1.0/(2.0/Draconic + 1.0/Anomalistic); -- nonlinear
   MfPrime : constant := 1.0/(1.0/Draconic + 1.0/Tropical); -- Mf'
   Nodal : constant := 1.0/(1.0/Draconic - 1.0/Tropical); -- Nodal
   Mt  : constant := 1.0/(2.0/Tropical + 1.0/Anomalistic); -- long alias
   MtPrime  : constant := 1.0/(1.0/MfPrime + 1.0/Anomalistic); -- long alias
   Absidal : constant := 1.0/(2.0/Year - 2.0/Tropical + 2.0/Anomalistic);
   Absidal2 : constant := 1.0/(2.0/Year - 2.0/Tropical + 2.0/Anomalistic + 1.0/Tropical - 1.0/Draconic);
   Evection : constant := 1.0/(2.0/Tropical - 1.0/Draconic);
   PerYear : constant := 20;
   dT : constant Long_Float := 1.0/Long_Float(PerYear);
   Start : constant := 1900.0;

function Filter_Moving_Average(Time : Float_Array; Values : Float_Array) return Float_Array is
   Output : Float_Array;
   Window_Size : constant Positive := 5;
   Sum : Long_Float;
   Count : Natural;
begin
   for I in Values'Range loop
      Sum   := 0.0;
      Count := 0;

      -- Look back up to 6 points (including current point)
      for J in 0 .. Window_Size - 1 loop
         if I - J >= Values'First then
            Sum   := Sum + Long_Float(Values(I - J));
            Count := Count + 1;
         end if;
      end loop;

      -- Calculate average for the current window
      Output(I) := Sum / Long_Float(Count);
   end loop;

   return Output;
end Filter_Moving_Average;
   

function Filter_Resonant(Time : Float_Array; Values : Float_Array) return Float_Array is
   Output : Float_Array;
   
   -- State variables
   Z1, Z2 : Long_Float := 0.0;
   
   -- Filter parameters
   Fc : constant Long_Float := 1.0;  -- Center frequency (1/year)
   Q  : constant Long_Float := 30.0; -- Quality factor
   
   -- Calculate Fs from the first two time steps
   -- (e.g., 1/12 for monthly, 1/20 for custom step)
   Delta_T : constant Long_Float := Long_Float(Time(Time'First + 1) - Time(Time'First));
   Fs      : constant Long_Float := 1.0 / Delta_T;
   
   -- Intermediate Variables for Bilinear Transform
   Pi     : constant Long_Float := Ada.Numerics.Pi;
   Omega  : constant Long_Float := 2.0 * Pi * Fc / Fs;
   Sin_W  : constant Long_Float := Sin(Omega);
   Cos_W  : constant Long_Float := Cos(Omega);
   Alpha  : constant Long_Float := Sin_W / (2.0 * Q);
   
   -- Coefficient Variables
   A0, A1, A2, B0, B1, B2 : Long_Float;
   W, X, Y : Long_Float;
begin
   -- Standard Bandpass Biquad Coefficients
   B0 := Alpha;
   B1 := 0.0;
   B2 := -Alpha;
   A0 := 1.0 + Alpha;
   A1 := -2.0 * Cos_W;
   A2 := 1.0 - Alpha;

   -- Normalize by A0
   B0 := B0 / A0;
   B1 := B1 / A0;
   B2 := B2 / A0;
   A1 := A1 / A0;
   A2 := A2 / A0;

   for I in Values'Range loop
      X := Long_Float(Values(I));

      -- Direct Form II Difference Equation
      W := X - (A1 * Z1) - (A2 * Z2);
      Y := (B0 * W) + (B1 * Z1) + (B2 * Z2);

      -- Update Delay Lines
      Z2 := Z1;
      Z1 := W;

      Output(I) := Y;
   end loop;

   return Output;
end Filter_Resonant;


   D, A, Total : Long_Float;
   
begin

   -- Values(Values'First) := 0.0;
   for I in Ordinal'Range loop
      Dates(I) := Start + Long_Float(I)*dT;
      D := sin(4.0*Pi*Year/Draconic*Dates(I)) + 0.01 * sin(2.0*Pi*Year/Draconic*Dates(I));
      A := sin(2.0*Pi*Year/Anomalistic*Dates(I)) -0.4*sin(4.0*Pi*Year/Anomalistic*Dates(I))  ;
      Total := D + 0.15 * A + 0.01*sin(2.0*Pi*Year/DA*Dates(I))
                            + 0.02*sin(2.0*Pi*Year/MfPrime*Dates(I))  
                            + 0.02*sin(2.0*Pi*Year/Nodal*Dates(I))  
                            + 0.01*sin(2.0*Pi*Year/Mt*Dates(I))  
                            + 0.01*cos(2.0*Pi*Year/MtPrime*Dates(I))  
                            + 0.005*sin(2.0*Pi*Year/Absidal*Dates(I))  
                            + 0.005*cos(2.0*Pi*Year/Absidal2*Dates(I))  
                            + 0.04*cos(4.0*Pi*Year/Evection*Dates(I))  
                            + 0.02*sin(2.0*Pi*Dates(I)) 
                            + 0.03*sin(4.0*Pi*Dates(I))
                            + 0.01*sin(4.0*Pi*Year/Tropical*Dates(I));
      if I mod (PerYear/2) = 0 then
         Values(I) := 1.0 * Total;
      elsif I mod (PerYear/2) = 1 or I mod (PerYear/2) = PerYear/2 - 1 then
         Values(I) := 0.8 * Total;
      elsif I mod (PerYear/2) = 2 or I mod (PerYear/2) = PerYear/2 - 2 then
         Values(I) := 0.5 * Total;
      else
         Values(I) := 0.0;
      end if;
      if I > Ordinal'First then
         Values(I) := Values(I-1) + Values(I); -- + 0.0025*sin(2.0*Pi*Dates(I)) + 0.01*sin(4.0*Pi*Dates(I));
      end if;
      -- Text_IO.Put_Line(Dates(I)'Img & " " & Values(I)'Img);
   end loop;
   
--   Values := Filter_Moving_Average(Dates, Filter_Moving_Average(Dates, Filter_Moving_Average(Dates, Values)));
--   Values := Filter_Resonant(Dates, Filter_Resonant(Dates, Filter_Resonant(Dates, Values)));
   Values := Filter_Moving_Average(Dates, Values);
   Values := Filter_Resonant(Dates, Values);
  
   
   for I in Ordinal'Range loop
      Text_IO.Put_Line(Dates(I)'Img & " " & Values(I)'Img);
   end loop;
   
end CW;

