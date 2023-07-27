import React, { useState } from "react";
import Logo from "../assets/logo.jpg";
import Searchbar from "./Searchbar";
import { Link } from "react-router-dom";
import ReorderIcon from "@material-ui/icons/Reorder";
import "../styles/Navbar.css";
import { Search } from "@material-ui/icons";

function Navbar() {
  const [openLinks, setOpenLinks] = useState(false);

  const toggleNavbar = () => {
    setOpenLinks(!openLinks);
  };
  return ( 
    <div className="navbar">
      <div className="leftSide" id={openLinks ? "open" : "close"}>
        <img src={Logo} />
        <Link to="/"> Main </Link>
        <Link to="/live"> Live </Link>
        <Link to="/alerts"> Alerts </Link>
        {/* <Link to="/home"> Contact </Link> */}
        {/* <Link to="/signin"> Sign In </Link> */}
      
        
        <div className="hiddenLinks">
          <Link to="/"> Main </Link>
          <Link to="/live"> Live </Link>
          <Link to="/alerts"> Alerts </Link>
          <Link to="/home"> Contact </Link>
          {/* <Link to="/signin"> Sign In </Link> */}
        </div>

        
      </div>
    
      <div className="rightSide" >
        <Searchbar ></Searchbar>
        
      </div>
    </div>
  );
}

export default Navbar;
