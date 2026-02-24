package Hal is

   pragma Elaborate_Body;

   -- convert form degrees to radians
   function Rads (Degrees : in Float) return Float;
   function Rads (Degrees : in Long_Float) return Long_Float;

   -- convert form radians to degrees
   function Degs (Radians : in Float) return Float;
   function Degs (Radians : in Long_Float) return Long_Float;

   ------------------------------------------------
   -- STRUCTURES - Basic geometry definitions
   ------------------------------------------------
   -- A,B,C are rotations about X,Y,Z respectively

   type Position is
      record
         X, Y, Z : Float;
      end record;
   pragma Convention (C, Position);

   type Position_Long is
      record
         X, Y, Z : Long_Float;
      end record;
   pragma Convention (C, Position_Long);


   function "+" (L, R : Position) return Position;
   function "*" (Scalar : Float; P : Position) return Position;
   function To_Str (Pos : Position) return String;
   function Terrain_To_Division_Cs (Terrain_Pos : Position) return Position;
   function Division_To_Terrain_Cs (Division_Pos : Position) return Position;

   type Orientation is
      record
         A, B, C : Float;
      end record;
   pragma Convention (C, Orientation);

   type Ori_Arr is array (Integer range <>) of Orientation;

   type Six_Dof is
      record
         Pos : Position;
         Ori : Orientation;
      end record;

   type Six_Dof_Arr is array (Positive range <>) of Six_Dof;

   function "+" (L, R : Orientation) return Orientation;
   function "*" (Scalar : Float; R : Orientation) return Orientation;
   function To_Str (Ori : Orientation; To_Degrees : Boolean := False) return String;
   function Terrain_To_Division_Cs (Terrain_Ori : Orientation) return Orientation;
   function Division_To_Terrain_Cs (Division_Ori : Orientation) return Orientation;

   type Axes is (X, Y, Z, A, B, C);
   function Get_Pos (Axis : Axes; Value : Float) return Position;
   function Get_Ori (Axis : Axes; Value : Float) return Orientation;
   function Get_Axis_Value (Axis : Axes; Pos : Position) return Float;
   function Get_Axis_Value (Axis : Axes; Ori : Orientation) return Float;

   -- Can indicate rate in terms of
   --   1. X Units / 1 second
   --   2. 1 Unit / X seconds
   ---
   type Rate is
      record
         Units : Float;
         Second : Integer := 1;
      end record;

   -- for magazine type operations
   type Rotation_Direction is (Cw, Ccw);

   -- for choosing which way to rotate.
   -- note that one of positive or negative will be the shortest_route
   type Direction_To_Rotate is (Pos, Neg, Shortest_Route);

   Max_Assembly_Name_Length : constant := 31;

   type Angle_Units is (Degrees, Radians, Mils, Yumamils);
   function Convert_Angle (Value : Float; From_Units : Angle_Units; To_Units : Angle_Units) return Float;

   -- $Id: hal.ads,v 1.12 2006/01/11 21:02:52 ludwiglj Exp $
end Hal;
--  This document contains trade secrets or other proprietary information.
--  It shall not be used, duplicated or disclosed in whole or in part
--  without the express written permission of United Defense, L.P.
--  This information is exempt from disclosure under Exemptions 3 and 4 of
--  the Freedom of Information Act. Any unauthorized disclosure by a
--  Government employee is a violation of the Trade Secrets Act. The printed
--  copy of the on-line document may not be current - verify current revision.

