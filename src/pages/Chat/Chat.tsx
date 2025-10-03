import ChatScreen from "../../components/ChatScreen";
import LogScreen from "../../components/LogScreen";
import ModelScreen from "../../components/ModelScreen";
import Navbar from "../../components/Navbar";
import Sidebar from "../../components/Sidebar";
import { useWllama } from "../../utils/wllama.context";
import { Screen } from "../../utils/types";
import GuideScreen from "../../components/GuideScreen";
export function Chat() {
  const { currScreen } = useWllama();

  return (
    <div className="flex flex-col drawer h-screen w-screen overflow-hidden">
      <Navbar />
      <div className="grow flex flex-row lg:drawer-open h-[calc(100vh-4rem)]">
        <Sidebar>
          {currScreen === Screen.MODEL && <ModelScreen />}
          {currScreen === Screen.CHAT && <ChatScreen />}
          {currScreen === Screen.GUIDE && <GuideScreen />}
          {currScreen === Screen.LOG && <LogScreen />}
        </Sidebar>
      </div>
    </div>
  );
}
