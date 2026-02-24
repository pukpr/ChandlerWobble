generic
   Key : in Integer;
   type Entities is (<>);     -- Enumerated type for all entities defined in the SDF
package HAL.Gazebo_Commands is

   -- Primary procedure to update a link's position
   -- Defaults allow calling only the fields you need to change
   procedure Set_Pose (
      Name  : in Entities;
      X     : in Long_Float := 0.0;
      Y     : in Long_Float := 0.0;
      Z     : in Long_Float := 0.0;
      Yaw   : in Long_Float := 0.0;
      Pitch : in Long_Float := 0.0;
      Roll  : in Long_Float := 0.0
   );

   -- Primary procedure to update a link's angular velocity
   -- Defaults allow calling only the fields you need to change
   procedure Set_Rot (
      Name  : in Entities;
      Yaw   : in Long_Float := 0.0;
      Pitch : in Long_Float := 0.0;
      Roll  : in Long_Float := 0.0
   );
   
   -- Primary procedure to apply a link's torque, this is not an impulse
   -- Defaults allow calling only the fields you need to change
   procedure Set_Torque (
      Name  : in Entities;
      X     : in Long_Float := 0.0;  -- these are offsets from the link base
      Y     : in Long_Float := 0.0;
      Z     : in Long_Float := 0.0;
      Yaw   : in Long_Float := 0.0;
      Pitch : in Long_Float := 0.0;
      Roll  : in Long_Float := 0.0
   );

end HAL.Gazebo_Commands;

