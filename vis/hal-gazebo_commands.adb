-- with Shared_Memory_Defs; use Shared_Memory_Defs;  
with Interfaces.C;       use Interfaces.C;
with Ada.Strings.Fixed;
with Pace.Keyed_Shared_Memory.Io;
with Ada.Characters.Handling;

package body Hal.Gazebo_Commands is

   package Shared_Memory_Defs is
      use Interfaces.C;

      -- Constants
      SHM_KEY      : constant int := 123456;
      MAX_ENTITIES : constant := 32;
      NAME_LEN     : constant := 64;

      -- Equivalent to EntityState struct
      type Entity_State is record
         Name     : char_array (0 .. NAME_LEN - 1);
         Command  : integer;
         X, Y, Z  : double;
         Roll     : double;
         Pitch    : double;
         Yaw      : double;
         Sequence : long_integer;
      end record
        with Convention => C;

      -- Ensure the Sequence field is treated as volatile/atomic
      --pragma Atomic (Entity_State.Sequence);
      --pragma Volatile_Components (Entity_State);

      -- Array type for the entities table
      type Entity_Array is array (0 .. MAX_ENTITIES - 1) of Entity_State
        with Convention => C;

      -- Equivalent to SharedWorldTable struct
      type Shared_World_Table is record
         Active_Entities : Integer;
         Entities        : Entity_Array;
      end record
        with Convention => C;

   end Shared_Memory_Defs;

   -- Static sequence counter
   Global_Sequence : Long_Integer := 0;

   subtype Table is Shared_Memory_Defs.Shared_World_Table;
   package Sm is new Pace.Keyed_Shared_Memory.Io (Key, -- 16#8bc# = 123456,
                                                  Table);
   
   -- Helper to copy Ada String to a fixed-size C char_array with null padding
   procedure Copy_Name_To_Buffer (Source : in String; Dest : out char_array) is
      C_Str : constant char_array := To_C(Source, Append_Nul => True);
   begin
      -- 1. Initialize destination with all nuls (Zero-out the buffer)
      Dest := (others => nul);

      -- 2. Copy the C_Str into the start of the buffer
      -- We must ensure we don't exceed the buffer size
      for I in C_Str'Range loop
         if (I - C_Str'First) <= Dest'Last then
            Dest(I - C_Str'First) := C_Str(I);
         end if;
      end loop;
   end Copy_Name_To_Buffer;
   
    
   -- We assume a pointer to the shared memory is available or globally defined
   -- For this example, we'll assume a procedure exists to get the table reference
   -- or that you will map the 'Table' pointer here.

   procedure Set_Pose (
      Name  : in Entities;
      X     : in Long_Float := 0.0;
      Y     : in Long_Float := 0.0;
      Z     : in Long_Float := 0.0;
      Yaw   : in Long_Float := 0.0;
      Pitch : in Long_Float := 0.0;
      Roll  : in Long_Float := 0.0
   ) is
      Entity_Idx : constant Integer := Entities'Pos(Name);
      C_Name : string := Ada.Characters.Handling.To_Lower (Entities'Image (Name));
   begin
      -- Increment sequence for every call to trigger the Gazebo plugin
      Global_Sequence := Global_Sequence + 1;
      
      Copy_Name_To_Buffer(C_Name, Sm.Value.Entities(Entity_Idx).Name);
      Sm.Value.Entities(Entity_Idx).Command := 0;
      Sm.Value.Entities(Entity_Idx).X := double(X);
      Sm.Value.Entities(Entity_Idx).Y := double(Y);
      Sm.Value.Entities(Entity_Idx).Z := double(Z);
      Sm.Value.Entities(Entity_Idx).Yaw := double(Yaw);
      Sm.Value.Entities(Entity_Idx).Pitch := double(Pitch);
      Sm.Value.Entities(Entity_Idx).Roll := double(Roll);
      Sm.Value.Entities(Entity_Idx).Sequence := Global_Sequence;
      Sm.Value.Active_Entities := Entities'Range_Length;
      
      -- Logic to write to your Shared_World_Table goes here:
      -- Table.Entities(Entity_Idx).X := double(X);
      -- Table.Entities(Entity_Idx).Y := double(Y);
      -- ...
      -- Table.Entities(Entity_Idx).Sequence := Global_Sequence;
   end Set_Pose;

   procedure Set_Rot (
      Name  : in Entities;
      Yaw   : in Long_Float := 0.0;
      Pitch : in Long_Float := 0.0;
      Roll  : in Long_Float := 0.0
   ) is
      Entity_Idx : constant Integer := Entities'Pos(Name);
      C_Name : string := Ada.Characters.Handling.To_Lower (Entities'Image (Name));
   begin
      -- Increment sequence for every call to trigger the Gazebo plugin
      Global_Sequence := Global_Sequence + 1;
      
      Copy_Name_To_Buffer(C_Name, Sm.Value.Entities(Entity_Idx).Name);
      Sm.Value.Entities(Entity_Idx).Command := 1;
      Sm.Value.Entities(Entity_Idx).Yaw := double(Yaw);
      Sm.Value.Entities(Entity_Idx).Pitch := double(Pitch);
      Sm.Value.Entities(Entity_Idx).Roll := double(Roll);
      Sm.Value.Entities(Entity_Idx).Sequence := Global_Sequence;
      Sm.Value.Active_Entities := Entities'Range_Length;
      
      -- Logic to write to your Shared_World_Table goes here:
      -- Table.Entities(Entity_Idx).X := double(X);
      -- Table.Entities(Entity_Idx).Y := double(Y);
      -- ...
      -- Table.Entities(Entity_Idx).Sequence := Global_Sequence;
   end Set_Rot;

   procedure Set_Torque (
      Name  : in Entities;
      X     : in Long_Float := 0.0;
      Y     : in Long_Float := 0.0;
      Z     : in Long_Float := 0.0;
      Yaw   : in Long_Float := 0.0;
      Pitch : in Long_Float := 0.0;
      Roll  : in Long_Float := 0.0
   ) is
      Entity_Idx : constant Integer := Entities'Pos(Name);
      C_Name : string := Ada.Characters.Handling.To_Lower (Entities'Image (Name));
   begin
      -- Increment sequence for every call to trigger the Gazebo plugin
      Global_Sequence := Global_Sequence + 1;
      
      Copy_Name_To_Buffer(C_Name, Sm.Value.Entities(Entity_Idx).Name);
      Sm.Value.Entities(Entity_Idx).Command := 2;
      Sm.Value.Entities(Entity_Idx).X := double(X);
      Sm.Value.Entities(Entity_Idx).Y := double(Y);
      Sm.Value.Entities(Entity_Idx).Z := double(Z);
      Sm.Value.Entities(Entity_Idx).Yaw := double(Yaw);
      Sm.Value.Entities(Entity_Idx).Pitch := double(Pitch);
      Sm.Value.Entities(Entity_Idx).Roll := double(Roll);
      Sm.Value.Entities(Entity_Idx).Sequence := Global_Sequence;
      Sm.Value.Active_Entities := Entities'Range_Length;
      
      -- Logic to write to your Shared_World_Table goes here:
      -- Table.Entities(Entity_Idx).X := double(X);
      -- Table.Entities(Entity_Idx).Y := double(Y);
      -- ...
      -- Table.Entities(Entity_Idx).Sequence := Global_Sequence;
   end Set_Torque;

end Hal.Gazebo_Commands;

