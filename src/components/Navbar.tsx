export default function Navbar() {
  return (
    <div className="navbar bg-gradient-to-r from-sky-900 via-sky-800 to-orange-500 text-white shadow-lg border-b border-sky-700 z-40">
  {/* Left Section (Drawer / Menu Icon) */}
  <div className="flex-none">
    <label
      htmlFor="my-drawer-2"
      className="btn btn-square btn-ghost flex lg:hidden hover:bg-sky-800"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        className="inline-block h-6 w-6 stroke-current"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M4 6h16M4 12h16M4 18h16"
        ></path>
      </svg>
    </label>
  </div>

  {/* Center Section (Logo + Title) */}
  <div className="flex-1 items-center gap-2 px-2 lg:mx-4">
    {/* Logo Icon */}
    <div className="flex items-center justify-center bg-orange-500 rounded-md p-1.5 shadow-md">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M13 16h-1v-4h-1m0 0V9a4 4 0 018 0v3m-4 4v1m0 4h-1m1 0h1"
        />
    </div>

    {/* Brand Name */}
    <a className="btn btn-ghost normal-case text-xl font-bold hover:bg-sky-800">
      kam<span className="text-orange-400">BRIng</span>
    </a>
  </div>

  {/* Right Section (optional user menu or icon) */}
  <div className="flex-none">
    <button className="btn btn-ghost btn-circle hover:bg-sky-800">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-9.33-4.993M13 21h-2a2 2 0 01-2-2v-1h6v1a2 2 0 01-2 2z"
        ></path>
    </button>
  </div>
</div>

    // <div className="navbar bg-base-100 shadow-md z-40">
    //   <div className="flex-none">
    //     <label
    //       htmlFor="my-drawer-2"
    //       className="btn btn-square btn-ghost flex lg:hidden"
    //     >
    //       <svg
    //         xmlns="http://www.w3.org/2000/svg"
    //         fill="none"
    //         viewBox="0 0 24 24"
    //         className="inline-block h-5 w-5 stroke-current"
    //       >
    //         <path
    //           strokeLinecap="round"
    //           strokeLinejoin="round"
    //           strokeWidth="2"
    //           d="M4 6h16M4 12h16M4 18h16"
    //         ></path>
    //       </svg>
    //     </label>
    //   </div>
    //   <a className="btn btn-ghost text-xl">kamBRIng</a>
    // </div>
  );
}
