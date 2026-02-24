with Ada.Text_Io;
with Ada.Tags;
with Ada.Unchecked_Deallocation;
with Ada.Environment_Variables;

package body Pace is

   function Flow (Message : in Msg'Class) return Channel is
   begin
      return new Msg'Class'(Message);
   end Flow;

   procedure Free_Channel is new Ada.Unchecked_Deallocation
                                   (Object => Msg'Class, Name => Channel);

   procedure Free (Obj : in out Channel) is
   begin
      Free_Channel (Obj);
   end Free;

   function Tag (Message : in Msg'Class) return String is
   begin
      return Ada.Tags.External_Tag (Message'Tag);
   end Tag;

   -- These would normally be overridden
   procedure Input (Message : in Msg) is
   begin
      Ada.Exceptions.Raise_Exception
        (Dispatch_Error'Identity,
         "called primitive " & Ada.Tags.External_Tag (Msg'Class (Message)'Tag));
      --Error ("No Input dispatching: " &
      --     Ada.Tags.External_Tag (Msg'Class (Message)'Tag));
   end Input;

   procedure Inout (Message : in out Msg) is
   begin
      Ada.Exceptions.Raise_Exception
        (Dispatch_Error'Identity,
         "called primitive " & Ada.Tags.External_Tag (Msg'Class (Message)'Tag));
      --Error ("No InOut dispatching: " &
      --     Ada.Tags.External_Tag (Msg'Class (Message)'Tag));
   end Inout;

   procedure Output (Message : out Msg) is
   begin
      --Message := Message; -- suppress warnings
      Ada.Exceptions.Raise_Exception
        (Dispatch_Error'Identity,
         "called primitive " & Ada.Tags.External_Tag (Msg'Class (Message)'Tag));
      --Error ("No Output dispatching: " &
      --     Ada.Tags.External_Tag (Msg'Class (Message)'Tag));
   end Output;

   function Getenv (Name : in String; Default : in String) return String is
   begin
      if Ada.Environment_Variables.Exists (Name) then
         return Ada.Environment_Variables.Value (Name);
      else
         return Default;
      end if;
   end Getenv;

   function Getenv (Name : in String; Default : in Integer) return Integer is
   begin
      return Integer'Value (Getenv (Name, Integer'Image (Default)));
   end Getenv;

   function Getenv (Name : in String; Default : in Float) return Float is
   begin
      return Float'Value (Getenv (Name, Float'Image (Default)));
   end Getenv;

   --
   -- Error
   --
   procedure Error (Text : in String; X_Info : in String := "") is
   begin
      Ada.Text_Io.Put_Line (Ada.Text_Io.Current_Error,
                        "PACE-ERROR: " & Text & " : " & X_Info);
   end Error;

   --
   -- Display string to standard output.
   --
   Pace_Display : constant Boolean := Getenv ("PACE_DISPLAY", 1) = 1;

   procedure Display (Text : in String) is
   begin
      if Pace_Display then
         Ada.Text_Io.Put_Line ("PACE-INFO:" & Text);
      end if;
   end Display;

   This_Node : constant Node_Slot := Getenv ("PACE_NODE", 0);

   function Get return Node_Slot is
   begin
      return This_Node;
   end Get;

   function Is_Local (Node : Node_Slot) return Boolean is
   begin
      return Node = Get or Node = 0 or This_Node = 0;
   end Is_Local;

   procedure Initialize (Object : in out Channel_Msg) is
   begin
      Object.Reference := Null_Channel_Msg.Reference;
   end Initialize;

   procedure Adjust (Object : in out Channel_Msg) is
   begin
      if Object.Reference /= Null_Msg'Access then
         Object.Reference := new Msg'Class'(Object.Reference.all);
      end if;
   end Adjust;

   procedure Finalize (Object : in out Channel_Msg) is
   begin
      --  Note: Don't try to free statically allocated null msg
      if Object.Reference /= Null_Msg'Access then
         Free (Object.Reference);
      end if;
   end Finalize;


   function To_Channel_Msg (Obj : Msg'Class) return Channel_Msg is
      Result : Channel_Msg;
   begin
      Result.Reference := Flow (Obj);
      return Result;
   end To_Channel_Msg;

   function To_Msg (Obj : Channel_Msg) return Msg'Class is
   begin
      return Obj.Reference.all;
   end To_Msg;

   function To_Callback (Obj : Msg'Class) return Channel_Msg is
      Result : Channel_Msg := To_Channel_Msg (Obj);
   begin
      Result.Reference.Slot := Get;
      return Result;
   end To_Callback;

   procedure Write (Stream : access Ada.Streams.Root_Stream_Type'Class;
                    Item : in Channel_Msg) is
   begin
      Msg'Class'Output (Stream, Item.Reference.all);
   end Write;

   procedure Read (Stream : access Ada.Streams.Root_Stream_Type'Class;
                   Item : out Channel_Msg) is
   begin
      Item.Reference := Flow (Msg'Class'Input (Stream));
   end Read;

   package body Dispatching is

      Tr : Trace_Call := null;

      Ip : Input_Call := null;
      procedure Input (Obj : in Msg'Class) is
      begin
         if Ip = null then
            Pace.Input (Obj);  -- redispatches to correct Input
         else
            Ip (Obj);
         end if;
         if Tr /= null then
            Tr (Obj);
         end if;
      end Input;

      Io : Inout_Call := null;
      procedure Inout (Obj : in out Msg'Class) is
      begin
         if Io = null then
            Pace.Inout (Obj);  -- redispatches to correct InOut
         else
            Io (Obj);
         end if;
         if Tr /= null then
            Tr (Obj);
         end if;
      end Inout;

      Op : Output_Call := null;
      procedure Output (Obj : out Msg'Class) is
      begin
         if Op = null then
            Pace.Output (Obj);  -- redispatches to correct Output
         else
            Op (Obj);
         end if;
         if Tr /= null then
            Tr (Obj);
         end if;
      end Output;

      procedure Set_Input_Call (Call : in Input_Call) is
      begin
         Ip := Call;
      end Set_Input_Call;
      procedure Set_Inout_Call (Call : in Inout_Call) is
      begin
         Io := Call;
      end Set_Inout_Call;
      procedure Set_Output_Call (Call : in Output_Call) is
      begin
         Op := Call;
      end Set_Output_Call;
      procedure Set_Trace_Call (Call : in Trace_Call) is
      begin
         Tr := Call;
      end Set_Trace_Call;

   end Dispatching;

   Now_Clock : Clock_Call := null;
   Start_Time : Ada.Real_Time.Time := Ada.Real_Time.Clock;

   function Now return Duration is
      use Ada.Real_Time;
   begin
      if Now_Clock = null then
         return To_Duration (Clock - Start_Time);
      else
         return Now_Clock.all;
      end if;
   end Now;
   procedure Set_Clock (Clock : in Clock_Call) is
   begin
      Now_Clock := Clock;
   end Set_Clock;


   use Ada.Real_Time;


   function Get_Node (Obj : Msg'Class) return Node_Slot is
   begin
      return Obj.Slot;
   end Get_Node;

   procedure Set_Node (Obj : in out Msg'Class; Node : in Node_Slot) is
   begin
      Obj.Slot := Node;
   end Set_Node;

   function Get_Time (Obj : Msg'Class) return Duration is
   begin
      return To_Duration (Obj.Time);
   end Get_Time;

   function Get_Wait (Obj : Msg'Class) return Duration is
   begin
      return To_Duration (Obj.Wait);
   end Get_Wait;

   procedure Set_Time (Obj : in out Msg'Class) is
   begin
      Obj.Time := To_Time_Span (Pace.Now);
   end Set_Time;

   procedure Set_Wait (Obj : in out Msg'Class; Wait : Duration) is
   begin
      Obj.Wait := To_Time_Span (Wait);
   end Set_Wait;

   procedure Set_Async (Obj : in out Msg'Class) is
   begin
      Obj.Send := Async;
   end Set_Async;

------------------------------------------------------------------------------
-- $id: pace.adb,v 1.3 11/24/2003 21:53:53 ludwiglj Exp $
------------------------------------------------------------------------------
end Pace;
