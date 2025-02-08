import React, { useState } from "react";
import '../index.css';
import SubmitButton from "./SubmitButton";
import { InputContainer } from "./Styles/Container/InputContainer.style";

const KeywordInput = () => {
  const [keyword, setKeyword] = useState(""); // 키워드 상태 관리

  const handleInputChange = (e) => {
    setKeyword(e.target.value); // 입력값을 상태에 저장
  };

  const handleSubmit = () => {
    console.log("입력된 키워드:", keyword); // 키워드를 출력 (백엔드로 전달 예정)
    // 여기서 API 호출 등 추가 작업을 수행할 수 있음
  };

  return (
    <InputContainer>
      <h2>최신 뉴스와 관련된 키워드를 입력하고, 요약 영상을 만들어보세요!</h2>
      <div style={{ display: "flex", flexDirection: "row", border: "2px solid red" }}>
        <input
          type="text"
          placeholder="키워드를 입력하세요"
          value={keyword}
          onChange={handleInputChange}
          style={{
            padding: "10px",
            width: "80%",
            marginRight: "10px",
            borderRadius: "5px",
            border: "1px solid #ccc",
            position: "bottome-left"
          }}
        />
        <SubmitButton />
      </div>
    </InputContainer>
  );
};

export default KeywordInput;