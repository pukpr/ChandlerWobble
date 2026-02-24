with Ada.Task_Identification;
with Ada.Exceptions;
with Ada.Streams;
with Ada.Finalization;
with Ada.Real_Time;

package Pace is
   --------------------------------------------
   -- PACE -- Base types and utilities
   --------------------------------------------
   pragma Elaborate_Body;

   subtype Node_Slot is Integer;
   -- Each simulation runs in it own unique node.
   -- Local simulation is Node_Slot=0.

   type Msg is tagged private;

   -- Primitives for message dispatching
   procedure Input (Message : in Msg);
   procedure Inout (Message : in out Msg);
   procedure Output (Message : out Msg);
   Dispatch_Error : exception;  -- Should not call these directly

   --------------------------------------------------
   -- Converts access to msg to streamable
   --------------------------------------------------
   type Channel_Msg is private;

   Null_Channel_Msg : constant Channel_Msg;

   function To_Channel_Msg (Obj : Msg'Class) return Channel_Msg;
   function "+" (Obj : Msg'Class) return Channel_Msg renames To_Channel_Msg;
   -- The following keeps the Node_Slot local
   function To_Callback (Obj : Msg'Class) return Channel_Msg;

   function To_Msg (Obj : Channel_Msg) return Msg'Class;
   function "+" (Obj : Channel_Msg) return Msg'Class renames To_Msg;


   package Dispatching is
      procedure Input (Obj : in Msg'Class);
      procedure Inout (Obj : in out Msg'Class);
      procedure Output (Obj : out Msg'Class);
      -- Over-rides for dispatch
      type Input_Call is access procedure (Obj : in Msg'Class);
      procedure Set_Input_Call (Call : in Input_Call);
      type Inout_Call is access procedure (Obj : in out Msg'Class);
      procedure Set_Inout_Call (Call : in Inout_Call);
      type Output_Call is access procedure (Obj : out Msg'Class);
      procedure Set_Output_Call (Call : in Output_Call);
      -- Decorate with tracing
      type Trace_Call is access procedure (Obj : in Msg'Class);
      procedure Set_Trace_Call (Call : in Trace_Call);
   end Dispatching;

   -- Current time relative from start of execution
   function Now return Duration;

   -- Message External Tag name functions
   function Tag (Message : in Msg'Class) return String;

   -- Convenience renames for task_id's
   subtype Thread is Ada.Task_Identification.Task_Id;
   function Current return Thread renames Ada.Task_Identification.Current_Task;
   function Image (T : Ada.Task_Identification.Task_Id := Current) return String
         renames Ada.Task_Identification.Image;

   -- Retrieves the environment variable data associated with NAME.
   -- DEFAULT is returned if not existent.
   function Getenv (Name : in String; Default : in String) return String;
   function Getenv (Name : in String; Default : in Integer) return Integer;
   function Getenv (Name : in String; Default : in Float) return Float;

   function Get_Time (Obj : Msg'Class) return Duration;
   function Get_Wait (Obj : Msg'Class) return Duration;
   -- sets the time to pace.now
   procedure Set_Time (Obj : in out Msg'Class);
   procedure Set_Wait (Obj : in out Msg'Class; Wait : Duration);

   procedure Set_Async (Obj : in out Msg'Class);

   function Get_Node (Obj : Msg'Class) return Node_Slot;
   procedure Set_Node (Obj : in out Msg'Class; Node : in Node_Slot);

private

   -----------------------------------------------------
   -- NODES -- Logical and symbolic naming of nodes
   -----------------------------------------------------
   -- Executing Paces are attached to logical nodes
   -- returning an integer and symbolic nodes mapped to
   -- the logical node (returning a string)
   -- : Returns the logical node
   -- : Host name
   -- : Nodes file name
   function Get return Node_Slot;

   -- Type of message synchronization desired
   type Synchronization is (Sync,     -- Synchronized
                            Async,    -- Asynchronous
                            Simple,   -- Local call (No thread interaction)
                            Balk,     -- Suspend call
                            Timeout); -- Signal to un-suspend call

   type Delivery is (Default, One_Way, Two_Way, Reply);

   package Art renames Ada.Real_Time;
   Zero : Art.Time_Span renames Art.Time_Span_Zero;

   -- Base type for message passing, tagged
   type Msg is tagged
      record
         Slot : Node_Slot := 0;         -- Node identifier
         Id : Thread := Current;      -- Identifies calling task
         Send : Synchronization := Simple;
         Enum : Delivery := One_Way; -- Reserved for message dispatching
         Time : Art.Time_Span := Art.To_Time_Span (Now);
         Wait : Art.Time_Span := Zero;
      end record;

   -- Print error message to console
   procedure Error (Text : in String; X_Info : in String := "");
   function X_Info (X : in Ada.Exceptions.Exception_Occurrence) return String
     renames Ada.Exceptions.Exception_Information;

   -- Display string to standard output
   procedure Display (Text : in String);

   function Is_Local (Node : Node_Slot) return Boolean;

   -- Bus simulation type
   type Channel is access all Msg'Class;
   function Flow (Message : in Msg'Class) return Channel;
   procedure Free (Obj : in out Channel);

   package Af renames Ada.Finalization;

   Null_Msg : aliased Msg := (0, Current, Simple, One_Way, Zero, Zero);

   type Channel_Msg is new Af.Controlled with
      record
         Reference : Channel := Null_Msg'Access;
      end record;

   procedure Initialize (Object : in out Channel_Msg);
   procedure Adjust (Object : in out Channel_Msg);
   procedure Finalize (Object : in out Channel_Msg);

   procedure Write (Stream : access Ada.Streams.Root_Stream_Type'Class;
                    Item : in Channel_Msg);
   for Channel_Msg'Write use Write;

   procedure Read (Stream : access Ada.Streams.Root_Stream_Type'Class;
                   Item : out Channel_Msg);
   for Channel_Msg'Read use Read;

   Null_Channel_Msg : constant Channel_Msg :=
     (Af.Controlled with Reference => Null_Msg'Access);

   type Clock_Call is access function return Duration;
   procedure Set_Clock (Clock : in Clock_Call);

   -- $Id: pace.ads,v 1.6 2005/11/10 22:24:03 pukitepa Exp $

end Pace;
--  This document contains trade secrets or other proprietary information.
--  It shall not be used, duplicated or disclosed in whole or in part
--  without the express written permission of United Defense, L.P.
--  This information is exempt from disclosure under Exemptions 3 and 4 of
--  the Freedom of Information Act. Any unauthorized disclosure by a
--  Government employee is a violation of the Trade Secrets Act. The printed
--  copy of the on-line document may not be current - verify current revision.
