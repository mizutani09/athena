// #ifndef TASK_LIST_NR_TASK_LIST_HPP_
// #define TASK_LIST_NR_TASK_LIST_HPP_
// //========================================================================================
// // Athena++ astrophysical MHD code
// // Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// // Licensed under the 3-clause BSD License, see LICENSE file for details
// //========================================================================================
// //! \file nr_task_list.hpp
// //! \brief define NewtonRaphsonTaskList class

// // C headers

// // C++ headers

// // Athena++ headers
// #include "../athena.hpp"
// #include "./task_list.hpp"

// // forward declarations
// class Mesh;
// class MeshBlock;
// class NewtonRaphson;
// class NewtonRaphsonDriver;
// class NewtonRaphsonTaskList;

// //----------------------------------------------------------------------------------------
// //! \struct NRTask
// //! \brief data and function pointer for an individual NRTask

// struct NRTask {
//   TaskID task_id;      //!> encodes task using bit positions in NewtonRaphsonTaskNames
//   TaskID dependency;   //!> encodes dependencies to other tasks using NewtonRaphsonTaskNames
//   TaskStatus (NewtonRaphsonTaskList::*TaskFunc)(NewtonRaphson*);  //!> ptr to a task
// };


// //----------------------------------------------------------------------------------------
// //! \class NewtonRaphsonTaskList
// //! \brief data and function definitions for NewtonRaphsonTaskList class

// class NewtonRaphsonTaskList {
//  public:
//   explicit NewtonRaphsonTaskList(NewtonRaphsonDriver *pmd) : ntasks(0), pmy_nrdriver_(pmd),
//                                                      task_list_{} {}
//   // data
//   int ntasks;     //!> number of tasks in this list

//   // functions
//   TaskListStatus DoAllAvailableTasks(NewtonRaphson *pmg, TaskStates &ts);
//   void DoTaskListOneStage(NewtonRaphsonDriver *pmd);
//   void ClearTaskList() {ntasks=0;}

//   // functions
//   TaskStatus StartReceive(NewtonRaphson *pmg);
//   TaskStatus StartReceiveFluxCons(NewtonRaphson *pmg);
//   TaskStatus StartReceiveForProlongation(NewtonRaphson *pmg);
//   TaskStatus ClearBoundary(NewtonRaphson *pmg);
//   TaskStatus ClearBoundaryFluxCons(NewtonRaphson *pmg);
//   TaskStatus SendBoundaryFluxCons(NewtonRaphson *pmg);
//   TaskStatus SendBoundaryForProlongation(NewtonRaphson *pmg);
//   TaskStatus ReceiveBoundaryFluxCons(NewtonRaphson *pmg);
//   TaskStatus ReceiveBoundaryForProlongation(NewtonRaphson *pmg);
//   TaskStatus SmoothRed(NewtonRaphson *pmg);
//   TaskStatus SmoothBlack(NewtonRaphson *pmg);
//   TaskStatus PhysicalBoundary(NewtonRaphson *pmg);
//   TaskStatus Restrict(NewtonRaphson *pmg);
//   TaskStatus Prolongate(NewtonRaphson *pmg);
//   TaskStatus ProlongateBoundary(NewtonRaphson *pmg);
//   TaskStatus ProlongateBoundaryForProlongation(NewtonRaphson *pmg);
//   TaskStatus CalculateFASRHS(NewtonRaphson *pmg);
//   TaskStatus StoreOldData(NewtonRaphson *pmg);

//   void SetNRTaskListToFiner(int nsmooth, int ngh, int flag = 0);
//   void SetNRTaskListToCoarser(int nsmooth, int ngh);
//   void SetNRTaskListFNRProlongate(int flag = 0);
//   void SetNRTaskListBoundaryCommunication();

//  private:
//   NewtonRaphsonDriver* pmy_nrdriver_;
//   NRTask task_list_[64*TaskID::kNField_];

//   void AddNewtonRaphsonTask(const TaskID& id, const TaskID& dep);
// };

// //----------------------------------------------------------------------------------------
// //! 64-bit integers with "1" in different bit positions used to ID each NewtonRaphson task.

// namespace NewtonRaphsonTaskNames {
// const TaskID NONE(0);
// const TaskID NR_STARTRECV0(1);
// const TaskID NR_STARTRECV1R(2);
// const TaskID NR_STARTRECV1B(3);
// const TaskID NR_STARTRECV2R(4);
// const TaskID NR_STARTRECV2B(5);
// const TaskID NR_STARTRECVP(6);
// const TaskID NR_STARTRECVL(7);
// const TaskID NR_CLEARBND0(8);
// const TaskID NR_CLEARBND1R(9);
// const TaskID NR_CLEARBND1B(10);
// const TaskID NR_CLEARBND2R(11);
// const TaskID NR_CLEARBND2B(12);
// const TaskID NR_CLEARBNDP(13);
// const TaskID NR_CLEARBNDL(14);
// const TaskID NR_SENDBND0(15);
// const TaskID NR_SENDBND1R(16);
// const TaskID NR_SENDBND1B(17);
// const TaskID NR_SENDBND2R(18);
// const TaskID NR_SENDBND2B(19);
// const TaskID NR_SENDBNDP(20);
// const TaskID NR_SENDBNDL(21);
// const TaskID NR_RECVBND0(22);
// const TaskID NR_RECVBND1R(23);
// const TaskID NR_RECVBND1B(24);
// const TaskID NR_RECVBND2R(25);
// const TaskID NR_RECVBND2B(26);
// const TaskID NR_RECVBNDP(27);
// const TaskID NR_RECVBNDL(28);
// const TaskID NR_PRLNGBNDP(29);
// const TaskID NR_PRLNGFC0(30);
// const TaskID NR_PRLNGFC1R(31);
// const TaskID NR_PRLNGFC1B(32);
// const TaskID NR_PRLNGFC2R(33);
// const TaskID NR_PRLNGFC2B(34);
// const TaskID NR_PRLNGFCL(35);
// const TaskID NR_SMOOTH1R(36);
// const TaskID NR_SMOOTH1B(37);
// const TaskID NR_SMOOTH2R(38);
// const TaskID NR_SMOOTH2B(39);
// const TaskID NR_PHYSBND0(40);
// const TaskID NR_PHYSBND1R(41);
// const TaskID NR_PHYSBND1B(42);
// const TaskID NR_PHYSBND2R(43);
// const TaskID NR_PHYSBND2B(44);
// const TaskID NR_PHYSBNDL(45);
// const TaskID NR_RESTRICT(46);
// const TaskID NR_PROLONG(47);
// const TaskID NR_FNRPROLONG(48);
// const TaskID NR_CALCFASRHS(49);
// } // namespace NewtonRaphsonTaskNames

// #endif // TASK_LIST_NR_TASK_LIST_HPP_
