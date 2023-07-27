import "./App.css";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Searchbar from "./components/Searchbar";
import Home from "./pages/Home";
import Live from "./pages/Live";
import Alerts from "./pages/Alerts"
import About from "./pages/About";
import Contact from "./pages/Contact";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import SignInSide from "./pages/SignInSide";

function App() {
  return (
    <div className="App">
      <Router>
        <Navbar /> 
    
        <Switch>
          <Route path="/" exact component={Home} />
          <Route path="/live" exact component={Live} />
          <Route path="/alerts" exact component={Alerts} />
          {/* <Route path="/about" exact component={About} /> */}
          {/* <Route path="/contact" exact component={Contact} /> */}
          {/* <Route path="/signin" exact component={SignInSide} /> */}
          
        </Switch>
        <Footer />
      </Router>
    </div>
  );
}

export default App;
