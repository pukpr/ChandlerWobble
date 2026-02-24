with Ada.Numerics;

package body Hal is

   Pi_Factor : constant := Ada.Numerics.Pi / 180.0;

   function Rads (Degrees : in Float) return Float is
   begin
      return Degrees * Pi_Factor;
   end Rads;

   function Rads (Degrees : in Long_Float) return Long_Float is
   begin
      return Degrees * Pi_Factor;
   end Rads;

   function Degs (Radians : in Float) return Float is
   begin
      return Radians / Pi_Factor;
   end Degs;

   function Degs (Radians : in Long_Float) return Long_Float is
   begin
      return Radians / Pi_Factor;
   end Degs;

   Num_In_Circle : array (Angle_Units'Range) of Float := (Degrees => 360.0,
                                                          Radians => 2.0 * Ada.Numerics.Pi,
                                                          Mils => 6400.0,
                                                          Yumamils => 6400.0);

   function Convert_Angle (Value : Float; From_Units : Angle_Units; To_Units : Angle_Units) return Float is
   begin
      return Value * Num_In_Circle (To_Units) / Num_In_Circle (From_Units);
   end Convert_Angle;


   function "+" (L, R : Position) return Position is
   begin
      return (L.X + R.X, L.Y + R.Y, L.Z + R.Z);
   end "+";

   function "*" (Scalar : Float; P : Position) return Position is
   begin
      return (Scalar * P.X, Scalar * P.Y, Scalar * P.Z);
   end "*";

   function To_Str (Pos : Position) return String is
   begin
      return ("(" & Pos.X'Img & ", " & Pos.Y'Img &
              ", " & Pos.Z'Img & ")");
   end To_Str;


   function Terrain_To_Division_Cs (Terrain_Pos : Position) return Position is
   begin
      return (-Terrain_Pos.Y, -Terrain_Pos.Z, Terrain_Pos.X);
   end Terrain_To_Division_Cs;

   function Division_To_Terrain_Cs (Division_Pos : Position) return Position is
   begin
      return (Division_Pos.Z, -Division_Pos.X, -Division_Pos.Y);
   end Division_To_Terrain_Cs;


   function "+" (L, R : Orientation) return Orientation is
   begin
      return (L.A + R.A, L.B + R.B, L.C + R.C);
   end "+";

   function "*" (Scalar : Float; R : Orientation) return Orientation is
   begin
      return (Scalar * R.A, Scalar * R.B, Scalar * R.C);
   end "*";

   function To_Str (Ori : Orientation; To_Degrees : Boolean := False) return String is
   begin
      if To_Degrees then
         return ("(" & Hal.Degs (Ori.A)'Img & ", " &
                 Hal.Degs (Ori.B)'Img &
                 ", " & Hal.Degs (Ori.C)'Img & ")");
      else
         return ("(" & Ori.A'Img & ", " & Ori.B'Img &
                 ", " & Ori.C'Img & ")");
      end if;
   end To_Str;

   function Terrain_To_Division_Cs (Terrain_Ori : Orientation) return Orientation is
   begin
      return (-Terrain_Ori.B, -Terrain_Ori.C, Terrain_Ori.A);
   end Terrain_To_Division_Cs;

   function Division_To_Terrain_Cs (Division_Ori : Orientation) return Orientation is
   begin
      return (Division_Ori.C, -Division_Ori.A, -Division_Ori.B);
   end Division_To_Terrain_Cs;

   function Get_Pos (Axis : Axes; Value : Float) return Position is
      Pos : Position := (0.0, 0.0, 0.0);
   begin
      if Axis = X then
         Pos.X := Value;
      elsif Axis = Y then
         Pos.Y := Value;
      else
         Pos.Z := Value;
      end if;
      return Pos;
   end Get_Pos;

   function Get_Ori (Axis : Axes; Value : Float) return Orientation is
      Ori : Orientation := (0.0, 0.0, 0.0);
   begin
      if Axis = A then
         Ori.A := Value;
      elsif Axis = B then
         Ori.B := Value;
      else
         Ori.C := Value;
      end if;
      return Ori;
   end Get_Ori;

   function Get_Axis_Value (Axis : Axes; Pos : Position) return Float is
   begin
      if Axis = X then
         return (Pos.X);
      elsif Axis = Y then
         return (Pos.Y);
      else
         return (Pos.Z);
      end if;
   end Get_Axis_Value;

   function Get_Axis_Value (Axis : Axes; Ori : Orientation) return Float is
   begin
      if Axis = A then
         return (Ori.A);
      elsif Axis = B then
         return (Ori.B);
      else
         return (Ori.C);
      end if;
   end Get_Axis_Value;

end Hal;

