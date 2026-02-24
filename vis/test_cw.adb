--with Articulated_Arm;
--use Articulated_Arm;
with Hal.Gazebo_Commands;
with Ada.Numerics.Long_Elementary_Functions;

procedure Test_CW is
   use Ada.Numerics, Ada.Numerics.Long_Elementary_Functions;

 type Elements is (
      Earth,
      Moon,
      Sun,
      sphere_link,
      cylinder_link
   );
   package E is new Hal.Gazebo_Commands(123456, Elements);

   dT : constant := 0.03;
   
   task Wobble;
   task body Wobble is
    Time : Long_Float := 0.0; 
    Lunar_Torque : Long_Float := 1.0;
   begin
    for I in 1..1_000_00 loop
      E.Set_Torque(Name => Earth, Roll => Lunar_Torque*sin(2.0*pi*Time/5.0), 
                                  Pitch => Lunar_Torque*cos(2.0*pi*Time/5.0));
      delay 5.0*dT;
      Time := Time + 5.0*dT;
    end loop;    
   end Wobble;

   task Solar;
   task body Solar is
    Time : Long_Float := 0.0; 
    Annual : Long_Float := 30.001;
    Dip : Long_Float := 2.0;
   begin
    for I in 1..1_000_00 loop
      E.Set_Rot(Name => Sun, Yaw => Annual, Pitch => Dip*cos(2.0*pi*Time/5.0));
      delay 3.0*dT;
      Time := Time + dT;
    end loop;    
   end Solar;

   task Lunar;
   task body Lunar is
    Time : Long_Float := 0.0; 
    Draconic : Long_Float := 260.001;
    Dip : Long_Float := 100.0;
   begin
    for I in 1..1_000_00 loop
      E.Set_Rot(Name => Moon, Yaw => Draconic, Pitch => Dip*cos(2.0*pi*Time*0.35));
      -- E.Set_Pose(Name => Moon, Pitch => 0.1);
      delay 2.0*dT;
      Time := Time + dT;
    end loop;    
   end Lunar;

   Daily : Long_Float := 10000.001;

begin
   for I in 1..1_000_00 loop
      E.Set_Rot(Name => Earth, Yaw => Daily);
      delay dT;
   end loop;    
end;
